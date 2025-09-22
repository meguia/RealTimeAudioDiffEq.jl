include("libportaudio.jl")

using .LibPortAudio
using DifferentialEquations
using Atomix: @atomic, @atomicswap
using Base.Threads

mutable struct RBAudio
    buf::Vector{Float32}
    mask::Int
    w::Atomic{Int}
    r::Atomic{Int}
end

mutable struct RealTimeAudioDEControlData
	@atomic u0::Vector{Float64}
	@atomic p::Vector{Float64}
	@atomic ts::Float64
	@atomic gain::Float64
	@atomic channel_map::Matrix{Float64}
end

mutable struct RealTimeAudioDEStateData
	t::Float64
	u::Vector{Float64}
end

# Fill this upon stream creation (in rt_ODEStart)
mutable struct RealTimeAudioDEStreamData
	sample_rate::Float64
	buffer_size::UInt64
	n_channels::UInt64
	stream::Ref{Ptr{PaStream}}
end

mutable struct RealTimeAudioDEData
	problem::DEProblem
	alg::DEAlgorithm
	control::RealTimeAudioDEControlData
	state::RealTimeAudioDEStateData
	stream_data::RealTimeAudioDEStreamData
	 # --- NEW ---
    G::Matrix{Float64}                       # normalized gain matrix (nvars × n_channels)
    rb::Union{RBAudio, Nothing}              # ring buffer for inter-thread audio samples
    producer::Union{Task, Nothing}           # background integrator task
    integrator::Any                          # ODE/SDE integrator (init(...) result)
    running::Atomic{Bool}                    # run flag for producer
	# diagnostics (atomic counters)
    frames_cb::Atomic{Int}           # number of PortAudio callback frames processed
    samples_popped::Atomic{Int}      # total samples popped out to PortAudio
    underflows::Atomic{Int}          # times we had to zero-fill
    framesize_mismatch::Atomic{Int}  # times framesPerBuffer ≠ configured frames
    pushed_samples::Atomic{Int}      # total samples the producer pushed
end

#! export
mutable struct DESource
	data::RealTimeAudioDEData
	callback::Base.CFunction
end

function RBAudio(n::Integer)
    N = 1 << (ceil(Int, log2(max(n, 1024))))     # next power of two, >= 1024
    RBAudio(zeros(Float32, N), N-1, Atomic{Int}(0), Atomic{Int}(0))
end

@inline rb_capacity(rb::RBAudio) = length(rb.buf)
@inline rb_available_to_write(rb::RBAudio) = rb_capacity(rb) - (rb.w[] - rb.r[])
@inline rb_available_to_read(rb::RBAudio)  = rb.w[] - rb.r[]
@inline rb_is_empty(rb::RBAudio) = rb.w[] == rb.r[]
@inline rb_is_full(rb::RBAudio)  = rb_available_to_write(rb)

#! export
function audio_debug_status(src::DESource)
    d = src.data
    (
        frames_cb          = d.frames_cb[],
        samples_popped     = d.samples_popped[],
        pushed_samples     = d.pushed_samples[],
        underflows         = d.underflows[],
        framesize_mismatch = d.framesize_mismatch[],
        rb_level           = d.rb === nothing ? -1 : (d.rb.w[] - d.rb.r[]),
        rb_capacity        = d.rb === nothing ? -1 : rb_capacity(d.rb)
    )
end

# pop up to n samples; returns number popped
function rb_pop!(rb::RBAudio, out::Ptr{Cfloat}, n::Int)
    avail = rb.w[] - rb.r[]
    m = min(avail, n)
    if m <= 0
        return 0
    end
    ri   = rb.r[] & rb.mask
    cap  = length(rb.buf)
    tail = min(m, cap - ri)

    # first contiguous chunk
    src1 = pointer(rb.buf, ri + 1)                 # ::Ptr{Float32}
    Base.unsafe_copyto!(out, src1, tail)           # 3-arg form: (dest::Ptr, src::Ptr, nelems)

    if tail < m
        # wrapped chunk
        src2 = pointer(rb.buf, 1)
        Base.unsafe_copyto!(out + tail, src2, m - tail)
    end

    rb.r[] = rb.r[] + m
    return m
end

# push 1 sample (caller checks space)
@inline function rb_push!(rb::RBAudio, x::Float32)
    wi = rb.w[] & rb.mask
    @inbounds rb.buf[wi+1] = x
    rb.w[] = rb.w[] + 1
    nothing
end

function normalize_channel_map(channel_map, nvars::Int, nch::Int)::Matrix{Float64}
    G = zeros(Float64, nvars, nch)
    if channel_map isa Vector{Int}
        for (ch, var) in enumerate(channel_map[1:min(end, nch)])
            if 1 <= var <= nvars
                G[var, ch] = 1.0
            end
        end
    elseif channel_map isa Vector{Vector{Int}}
        for ch in 1:nch
            ch > length(channel_map) && break
            for var in channel_map[ch]
                if 1 <= var <= nvars
                    G[var, ch] += 1.0
                end
            end
        end
    elseif channel_map isa AbstractMatrix
        @inbounds for ch in 1:nch, k in 1:min(nvars, size(channel_map,1))
            G[k, ch] = Float64(channel_map[k, ch])
        end
    else
        error("Unsupported channel_map type")
    end
    return G
end

function rebuild_integrator_f64!(src::DESource)
    d   = src.data
    fs  = d.stream_data.sample_rate               # Float64
    dt  = (1.0 / fs) * d.control.ts               # Float64
    t0  = Float64(d.state.t)
    u0  = Float64.(d.state.u)
    p   = Float64.(d.control.p)

    rprob = remake(d.problem; u0=u0, tspan=(t0, Inf), p=p)

    d.integrator = init(rprob, d.alg; dt=dt, adaptive=false,
                 save_everystep=false, save_start=false, save_end=false)
end


# Peek at the next mixed sample without changing the audio path
function current_mix_snapshot(src::DESource)
    integ = src.data.integrator
    G = src.data.G
    nch = size(G,2); nvars = size(G,1)
    y = integ.u
    s = zeros(Float64, nch)
    @inbounds @simd for ch in 1:nch
        acc = 0.0
        @inbounds @simd for k in 1:nvars
            acc += G[k,ch]*y[k]
        end
        s[ch] = src.data.control.gain * acc
    end
    (; t = integ.t, sample = s)
end

#! export
# ODE
"""
    DESource(f, u0::Vector{Float64}, p::Vector{Float64};
		alg = Tsit5(), channel_map::Matrix{Float64})::DESource

Create a DESource from an ODEFunction.
# Arguments
- `f::ODEFunction`: the ODE function. Should be of the form `f(du, u, p, t)` (In-place).
- `u0`: the array of initial values.
- `p`: the array of parameters.
# Keyword Arguments
- `alg::DEAlgorithm = Tsit5()`: the algorithm which will be passed to the solver.
- `channel_map =  [1. 0.; 0. 1.]`: the channel map sized (nvars, nchannels) indicates \
how the variables of the system are mapped to the output channels. Each column is a channel, \
and each row is a variable. The values in the matrix are the gains for each variable in that channel
"""
function DESource(f, u0::Vector{Float64}, p::Vector{Float64};
		alg = Tsit5(), channel_map::Matrix{Float64})::DESource

	prob = ODEProblem(f, u0, (0.0, 0.01), p;  
		save_start = true,
		save_end = true, 
		verbose = false)

	_DESource(prob, alg, channel_map)
end

function DESource(f::ODEFunction, u0::Vector{Float64}, p::Vector{Float64};
	alg = Tsit5(), channel_map::Matrix{Float64})::DESource

prob = ODEProblem(f, u0, (0.0, 0.01), p;  
	save_start = true,
	save_end = true, 
	verbose = false)

_DESource(prob, alg, channel_map)
end

#! export
# SDE
"""
    DESource(f, g, u0::Vector{Float64}, p::Vector{Float64};
		alg = SOSRA(), channel_map::Matrix{Float64})::DESource

Create a Stochastic DESource from a drift function and a noise function.
"""
function DESource(f, g, u0::Vector{Float64}, p::Vector{Float64};
		alg = SOSRA(), channel_map::Matrix{Float64})::DESource

	prob = SDEProblem(f, g, u0, (0.0, 0.01), p; 
		save_start = true,
		save_end = true, 
		verbose = false)

	_DESource(prob, alg, channel_map)
end

function DESource(f::SDEFunction, u0::Vector{Float64}, p::Vector{Float64};
	alg = SOSRA(), channel_map::Matrix{Float64})::DESource

prob = SDEProblem(f, u0, (0.0, 0.01), p; 
	save_start = true,
	save_end = true, 
	verbose = false)

_DESource(prob, alg, channel_map)
end

function _DESource(prob::DEProblem, alg, channel_map)::DESource

	control = RealTimeAudioDEControlData(prob.u0, prob.p, 1., 1., channel_map)
	
	u = deepcopy(prob.u0)
	state = RealTimeAudioDEStateData(prob.tspan[1], u)

	callback = create_callback()

	callback_ptr = @cfunction($callback, PaStreamCallbackResult, 
				(Ptr{Cvoid}, 
				Ptr{Cvoid}, 
				Culong, 
				Ptr{PaStreamCallbackTimeInfo}, 
				PaStreamCallbackFlags, 
				Ptr{Cvoid}))

	stream = RealTimeAudioDEStreamData(-1, 0, 0, Ref{Ptr{PaStream}}(0))

	#data = RealTimeAudioDEData(prob, alg, control, state, stream, zeros(Float32, length(prob.u0), 1), 
	#nothing, nothing, nothing, Atomic{Bool}(false))
	data = RealTimeAudioDEData(prob, alg, control, state, stream, zeros(Float32, length(prob.u0), 1), 
		nothing, nothing, nothing, Atomic{Bool}(false), Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0))
	return DESource(data, callback_ptr)
end

function _produce_audio!(data::RealTimeAudioDEData)
    nch = size(data.G, 2)
    nvars = size(data.G, 1)
    rb = data.rb::RBAudio
    
    integ = data.integrator
    @assert integ !== nothing

	while data.running[]
        # Ensure there is space for nch samples (one frame)
        target = Int(floor(0.9 * rb_capacity(rb)))
        while rb_available_to_write(rb) >= nch && (rb.w[] - rb.r[]) < target
            step!(integ)  # requires f!(du,u,p,t) in-place and type-stable for good perf
			y = integ.u  # current state vector
			# Mix: sample for each output channel
			@inbounds @simd for ch in 1:nch
				s = 0.0
				@inbounds @simd for k in 1:nvars
					s += data.G[k, ch] * y[k]
				end
				rb_push!(rb, Float32(data.control.gain*s))
			end
			data.pushed_samples[] = data.pushed_samples[] + nch
        end
        Base.yield()
    end
    return
end

function create_callback()
	function callback(inputBuffer::Ptr{Cvoid}, 
			outputBuffer::Ptr{Cvoid}, 
			framesPerBuffer::Culong, 
			timeInfo::Ptr{PaStreamCallbackTimeInfo}, 
			statusFlags::PaStreamCallbackFlags, 
			userData::Ptr{Cvoid})::PaStreamCallbackResult

		data = unsafe_pointer_to_objref(userData)::RealTimeAudioDEData

		out_sample = convert(Ptr{Cfloat}, outputBuffer)
        frames = Int(framesPerBuffer)
        nch = Int(data.stream_data.n_channels)
        nsamp = frames * nch

		# count frames & detect mismatch
		data.frames_cb[] = data.frames_cb[] + 1
		expected = Int(data.stream_data.buffer_size)   # what we asked PA to use
		if expected != 0 && frames != expected
			data.framesize_mismatch[] = data.framesize_mismatch[] + 1
		end

        rb = data.rb
        if rb === nothing
            # Safety: fill silence if not ready
            @inbounds for i in 1:nsamp
                unsafe_store!(out_sample, 0.0f0, i)
            end
            return paContinue
        end

        # Pop as many samples as available; zero the rest
        popped = rb_pop!(rb, out_sample, nsamp)
		data.samples_popped[] = data.samples_popped[] + popped
        if popped < nsamp
			data.underflows[] = data.underflows[] + 1
            # zero-fill tail
            @inbounds for i in popped+1:nsamp
                unsafe_store!(out_sample, 0.0f0, i)
            end
        end

        return paContinue
    end
    return callback
end

#! export
"""
    start_DESource(source::DESource, output_device::PaDeviceIndex;
    	sample_rate::Float64 = -1., 
    	buffer_size::UInt32 = convert(UInt32, paFramesPerBufferUnspecified ))

Start a DESource with a given output device.
# Keyword arguments
- `sample_rate::Float64 = -1.`: the sample rate of the stream. If negative, the default \
sample rate of the output device will be used.
- `buffer_size::UInt32 = convert(UInt32, paFramesPerBufferUnspecified )`: the buffer size \
of the stream.
"""
function start_DESource(source::DESource, output_device::PaDeviceIndex;
		sample_rate::Float64 = -1., 
		buffer_size::UInt32 = convert(UInt32, paFramesPerBufferUnspecified ))
	
	r_stream = source.data.stream_data.stream
	
	stream_status = Pa_IsStreamActive(r_stream[])
	if stream_status == 1
		println("Stream already active")
		return
	elseif stream_status == 0
		Pa_CloseStream(r_stream[])
	end

	if Pa_GetDeviceInfo(output_device) == C_NULL
		error("invalid output device.")
	end
	output_device_info = unsafe_load(Pa_GetDeviceInfo(output_device))

	 # Determine n_channels from channel_map and clamp to device capability
    n_channels = if source.data.control.channel_map isa AbstractMatrix
        size(source.data.control.channel_map, 2)
    else
        length(source.data.control.channel_map)
    end
	if output_device_info.maxOutputChannels < n_channels
		@warn "output device has less channels than channel map"
		n_channels = output_device_info.maxOutputChannels
	end
	source.data.stream_data.n_channels = n_channels

	# Normalize channel map to a gain matrix G (nvars × nch)
    nvars = length(source.data.problem.u0)
    source.data.G = normalize_channel_map(source.data.control.channel_map, nvars, n_channels)

	stream_parameters = PaStreamParameters(
		output_device,
		n_channels,	# channels
		paFloat32,	# sample format
		output_device_info.defaultLowOutputLatency,	# suggested latency
		C_NULL)		# host API specific stream info

	
	sample_rate = sample_rate < 0. ? output_device_info.defaultSampleRate : sample_rate
	source.data.stream_data.sample_rate = sample_rate
	source.data.stream_data.buffer_size = buffer_size #! Unnecessary?
	
	# --- Build integrator ONCE with fixed dt ---
    dt = Float64( (1.0 / sample_rate) * source.data.control.ts )
    # Make an "infinite" tspan from current state.t
    t0 = source.data.state.t
    prob = source.data.problem
    r_prob = remake(prob; u0 = source.data.state.u, tspan = (t0, Inf), p = source.data.control.p)
    integ = init(r_prob, source.data.alg; dt=dt, adaptive=false,
                 save_everystep=false, save_start=false, save_end=false)
    source.data.integrator = integ

    # --- Ring buffer sized for at least ~ 4 buffers of audio ---
    frames = buffer_size == paFramesPerBufferUnspecified ? 256 : Int(buffer_size)
	buffer_size = UInt32(frames)
	source.data.stream_data.buffer_size = buffer_size
    rb_len = 8 * frames * n_channels  # some headroom
    source.data.rb = RBAudio(rb_len)

    # --- Start producer task ---
    source.data.running[] = true
    source.data.producer = @async try
        _produce_audio!(source.data)
    catch e
        source.data.running[] = false
        @error "Producer task crashed" exception=(e, catch_backtrace())
    end
	# --- Start PortAudio stream ---
	err = Pa_OpenStream(
		r_stream,
		C_NULL,			# input parameters
		Ref(stream_parameters),	# output parameters
		sample_rate,
		buffer_size,
		0,
		source.callback,
		pointer_from_objref(source.data)
	)
	
	 if err != 0
        source.data.running[] = false
        error(unsafe_string(Pa_GetErrorText(err)))
    end

    err = Pa_StartStream(r_stream[])
    if err != 0
        source.data.running[] = false
        error(unsafe_string(Pa_GetErrorText(err)))
    end
    println("Start stream")
end

#! export
"""
    stop_DESource(source::DESource)
Stop a DESource.
"""
function stop_DESource(source::DESource)
	r_stream = source.data.stream_data.stream

	stream_status = Pa_IsStreamStopped(r_stream[])
	if stream_status == 1
		println("Stream already stopped")
	else
        err = Pa_StopStream(r_stream[])
        if err != 0
            error(unsafe_string(Pa_GetErrorText(err)))
        end
        println("Stop stream")
    end
	# Stop producer
    if source.data.running[]
        source.data.running[] = false
    end
    if source.data.producer !== nothing
        try
            wait(source.data.producer)
        catch
            # ignore
        end
        source.data.producer = nothing
    end
    source.data.rb = nothing
end

#! export
"""
    isinitialized(source::DESource)::Bool
Check if a DESource is initialized (it has been called `start_DESource()` \
on it at least once).
"""
function isinitialized(source::DESource)::Bool
	return source.data.stream_data.stream[] != C_NULL
end

#! export
"""
    isactive(source::DESource)::Bool
Check if a DESource is active.
"""
function isactive(source::DESource)::Bool
	return Pa_IsStreamActive(source.data.stream_data.stream[]) == 1
end

#! export
"""
    isstopped(source::DESource)::Bool
Return true only if `stop_DESource()` has been called on `source` at least \
once.
"""
function isstopped(source::DESource)::Bool
	return Pa_IsStreamStopped(source.data.stream_data.stream[]) == 1
end

#! export
"""
    reset_state!(source::DESource)
Reset a DESource to initial conditions.
"""
function reset_state!(source::DESource)
	source.data.state.t = 0.
	source.data.state.u = deepcopy(source.data.problem.u0)
end

#! export
"""
    set_u0!(source::DESource, u::Vector{Float64})
Set the initial values of the system in the DESource.
"""
function set_u0!(source::DESource, u::Vector{Float64})
	if length(u) != length(source.data.control.u0)
		error("u0 has different length than the system.")
	end
	@atomic source.data.control.u0 = u
end

#! export
"""
    get_u0(source::DESource)
Retrieve the initial values of the system in the DESource.
"""
function get_u0(source::DESource)
	return source.data.control.u0
end

#! export
"""
    set_param!(source::DESource, index::Int, value::Float64)
Set a parameter of the system in the DESource.
"""
function set_param!(source::DESource, index::Int, value::Float64)
	if index > length(source.data.control.p) || index < 1
		error("index out of bounds.")
	end
	@atomicswap source.data.control.p[index] = value; value
end

#! export
"""
    getparam(source::DESource, index::Int)
Retrieve the value of a single system's parameter in the DESource.
"""
function getparam(source::DESource, index::Int)
	if index > length(source.data.control.p) || index < 1
		error("index out of bounds.")
	end
	return source.data.control.p[index]
end

#! export
"""
    get_params(source::DESource)
Retrieve the values of all system's parameters in the DESource.
"""
function get_params(source::DESource)
	return source.data.control.p
end

#! export
"""
    set_ts!(source::DESource, value::Float64)
Set the time scale of the DESource.
"""
function set_ts!(source::DESource, value::Float64)
	@atomic source.data.control.ts = value
end

#! export
"""
    get_ts(source::DESource)
Retrieve the time scale of the DESource.
"""
function get_ts(source::DESource)
	return source.data.control.ts
end

#! export
"""
    set_gain!(source::DESource, value::Float64)
Set the gain of the DESource.
"""
function set_gain!(source::DESource, value::Float64)
	@atomic source.data.control.gain = value
end

#! export
"""
    get_gain(source::DESource)
Retrieve the gain of the DESource.
"""
function get_gain(source::DESource)
	return source.data.control.gain
end

#! export
"""
    set_channelmap!(source::DESource, channel_map::Matrix{Float64})
Set the channel map of the DESource.
"""
function set_channelmap!(source::DESource, channel_map::Matrix{Float64})
	# check variables
	n_vars = length(source.data.problem.u0)::Int
	
	if size(channel_map, 1) > n_vars
		@warn "$(size(channel_map, 1) - n_vars) variable(s) out of bounds."
	end
		
	status = Pa_IsStreamActive(source.data.stream_data.stream[])
	if status == 1 # stream is running
		# check number of channels
		n_channels_in_map = size(channel_map, 2)
		if n_channels_in_map != source.data.stream_data.n_channels
			error("# of channels in channel map and # of channels in the stream don't match. Stop the source and try again.")
		end
	end

	@atomic source.data.control.channel_map = channel_map
end

#! export
"""
    get_channelmap(source::DESource)
Retrieve the channel map of the DESource.
"""
function get_channelmap(source::DESource)
	return source.data.control.channel_map
end

#! export
"""
    get_default_output_device()::PaDeviceIndex
Get the default audio output device index.
"""
function get_default_output_device()::PaDeviceIndex
	return Pa_GetDefaultOutputDevice()
end

#! export
"""
    get_device_info(device::PaDeviceIndex)::PaDeviceInfo
Report information about an audio device.
"""
function get_device_info(device::PaDeviceIndex)::PaDeviceInfo
	if Pa_GetDeviceInfo(device) == C_NULL
		error("invalid device index.")
	end
	return unsafe_load(Pa_GetDeviceInfo(device))
end

#! export
"""
    get_devices()
Return an array of available audio devices.
"""
function get_devices()
	devices = []
	for i in 0:Pa_GetDeviceCount() - 1
		device_info = unsafe_load(Pa_GetDeviceInfo(i))
		d = (index = convert(PaDeviceIndex, i), 
			name = unsafe_string(device_info.name),
			inputs = device_info.maxInputChannels,
			outputs = device_info.maxOutputChannels,
			sr = device_info.defaultSampleRate)
		push!(devices, d)
	end
	return devices
end

#! export
"""
    list_devices()
Print a list of available audio devices.
"""
function list_devices()
	for i in 0:Pa_GetDeviceCount() - 1
		device_info = unsafe_load(Pa_GetDeviceInfo(i))
		println("Device: $i, ", device_info)
	end
end

#! export
"""
    get_device_index(name::String)::PaDeviceIndex
Get the index of an audio device by its name.
"""
function get_device_index(name::String)::PaDeviceIndex
	for i in 0:Pa_GetDeviceCount() - 1
		device_info = unsafe_load(Pa_GetDeviceInfo(i))
		if unsafe_string(device_info.name) == name
			return i
		end
	end
	println("Device not found")
	return -1
end

function Base.show(io::IO, di::PaDeviceInfo)
	print(io, "Name: ", '\"', unsafe_string(di.name), '\"', ", ")
	print(io, "Inputs: ", di.maxInputChannels, ", ")
	print(io, "Outputs: ", di.maxOutputChannels, ", ")
	print(io, "SR: ", di.defaultSampleRate)
end

function Base.show(io::IO, source::DESource)
	print(io, "DESource:", '\n')
	print(io, "f: ", source.data.problem.f.f, '\n')
	if isa(source.data.problem, SDEProblem)
		print(io, "g: ", source.data.problem.g, '\n')
	end
	print(io, "u0: ", source.data.problem.u0, '\n')
	print(io, "p: ", source.data.problem.p, '\n')
	print(io, "ts: ", source.data.control.ts, '\n')
	print(io, "gain: ", source.data.control.gain, '\n')
	print(io, "channel_map: ", source.data.control.channel_map, '\n')
end


function probe_integrator!(src::DESource; n=1024, take=16)
    integ = src.data.integrator
    integ === nothing && error("Integrator not initialized.")
    vals = Vector{Float32}(undef, min(n,take))
    changed = false
    last = Float32(integ.u[1])
    for i in 1:n
        step!(integ)
        y1 = Float32(integ.u[1])
        if i ≤ length(vals); vals[i] = y1; end
        changed |= (y1 != last)
        last = y1
    end
    (; changed, first_vals = vals)
end

# Show dt, t, types (sanity)
integrator_info(src::DESource) = (
    t = src.data.integrator.t,
    dt = getfield(src.data.integrator, :dt),  # works for OrdinaryDiffEq integrators
    eltype_u = eltype(src.data.integrator.u),
    eltype_p = eltype(src.data.integrator.p)
)