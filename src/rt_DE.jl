include("libportaudio.jl")
using .LibPortAudio
import SciMLBase
using SciMLBase: AbstractDEAlgorithm, AbstractSciMLProblem, DDEProblem, ODEFunction, ODEProblem,
    SDEFunction, SDEProblem, remake, init, step!, reinit!, set_proposed_dt!,
    successful_retcode, check_error, add_tstop!
using OrdinaryDiffEqTsit5: Tsit5
using DelayDiffEq: MethodOfSteps
import DelayDiffEq
import OrdinaryDiffEqCore
using StochasticDiffEq: SOSRA
using Base.Threads: Atomic

# One writer per index; publish only after copying complete frames.
mutable struct RBAudio
    buf::Vector{Float32}
    mask::Int
    w::Atomic{Int}
    r::Atomic{Int}
end

# The type parameter lets Julia specialize the complete sample loop.
mutable struct AudioRenderer{I,H}
    integrator::I
    history::H
    parameters::Vector{Float64}
    sample_rate::Float64
    timescale::Float64
    substeps::Int
    failed::Bool
    sample::Vector{Float64}
end

# Setters publish copies; only the producer mutates the integrator.
mutable struct RealTimeAudioDEControlData
    @atomic u0::Vector{Float64}
    @atomic p::Vector{Float64}
    @atomic ts::Float64
    @atomic gain::Float64
    @atomic channel_map::Matrix{Float64}
    @atomic reset::Union{Nothing,Vector{Float64}}
    writer_lock::ReentrantLock
end

mutable struct RealTimeAudioDEStateData
    t::Float64
    u::Vector{Float64}
end

mutable struct RealTimeAudioDEStreamData
    sample_rate::Float64
    buffer_size::UInt64
    n_channels::UInt64
    stream::Ref{Ptr{PaStream}}
    ever_started::Bool
end

mutable struct DelayWindow
    max_delay::Float64
    start::Float64
    dt::Float64
    steps::Int
    abstol::Float64
    reltol::Float64
    capacity::Int
    dtmax::Float64
    reuse::Bool
end

struct ReuseHistory{H}
    initial::H
    capacity::Int
end
(h::ReuseHistory)(args...; kwargs...) = h.initial(args...; kwargs...)

mutable struct HistoryBuffer{T} <: AbstractVector{T}
    slots::Vector{T}
    head::Int
    count::Int
end

function HistoryBuffer(values::Vector{T}, capacity) where T
    slots = [deepcopy(first(values)) for _ in 1:capacity]
    for i in eachindex(values)
        slots[i] = deepcopy(values[i])
    end
    HistoryBuffer{T}(slots, 1, length(values))
end

Base.size(b::HistoryBuffer) = (b.count,)
Base.IndexStyle(::Type{<:HistoryBuffer}) = IndexLinear()
@inline _slot(b::HistoryBuffer, i) = (j = b.head+i-1; j > length(b.slots) ? j-length(b.slots) : j)
Base.@propagate_inbounds function Base.getindex(b::HistoryBuffer, i::Int)
    @boundscheck checkbounds(b, i)
    b.slots[_slot(b, i)]
end
Base.@propagate_inbounds function Base.setindex!(b::HistoryBuffer, x, i::Int)
    @boundscheck checkbounds(b, i)
    _copy_record!(b.slots[_slot(b, i)], x)
    b
end

_copy_record!(dest::Vector{Float64}, src::Vector{Float64}) = copyto!(dest, src)
function _copy_record!(dest::Vector{Vector{Float64}}, src::Vector{Vector{Float64}})
    previous = length(dest)
    resize!(dest, length(src))
    for j in eachindex(src)
        if j > previous || length(dest[j]) != length(src[j])
            dest[j] = copy(src[j])
        else
            copyto!(dest[j], src[j])
        end
    end
    dest
end

# The default append copies before push!; specialize only our own storage.
function DelayDiffEq.copyat_or_push!(b::HistoryBuffer, i::Int, x)
    1 <= i <= min(b.count+1, length(b.slots)) || throw(BoundsError(b, i))
    _copy_record!(b.slots[_slot(b, i)], x)
    b.count = max(b.count, i)
    nothing
end

function Base.resize!(b::HistoryBuffer, n::Integer)
    0 <= n <= length(b.slots) || throw(BoundsError(b, n))
    b.count = n
    n == 0 && (b.head = 1)
    b
end
Base.empty!(b::HistoryBuffer) = resize!(b, 0)

function Base.deleteat!(b::HistoryBuffer, indices::UnitRange{Int})
    isempty(indices) && return b
    first(indices) == 1 && last(indices) <= b.count || throw(BoundsError(b, indices))
    b.head = _slot(b, last(indices)+1)
    b.count -= length(indices)
    b
end

# DelayDiffEq 6.3.1 builds history internally. Only our marked problems use this adapter.
function DelayDiffEq.build_history_function(
        prob::DDEProblem{U,T,L,L2,IIP,P,F,H}, alg, rate, reltol, dv; kwargs...
    ) where {U,T,L,L2,IIP,P,F,H<:ReuseHistory}
    original = remake(prob; h=prob.h.initial)
    old = DelayDiffEq.build_history_function(original, alg, rate, reltol, dv; kwargs...).integrator
    ts = old.sol.t
    us = HistoryBuffer(old.sol.u, prob.h.capacity)
    ks = HistoryBuffer(old.sol.k, prob.h.capacity)
    choices = old.sol.alg_choice
    sizehint!(ts, prob.h.capacity)
    choices === nothing || sizehint!(choices, prob.h.capacity)
    interp = OrdinaryDiffEqCore.InterpolationData(old.sol.interp.f, us, ts, ks,
        choices, true, old.cache, old.differential_vars, false)
    sol = SciMLBase.build_solution(old.sol.prob, old.alg, ts, us; dense=true,
        k=ks, interp, alg_choice=choices, calculate_error=false, stats=old.sol.stats)
    saved = DelayDiffEq.HistoryODEIntegrator{typeof(old.alg),IIP,typeof(old.u),
        typeof(old.t),typeof(old.tdir),typeof(old.k),typeof(sol),typeof(old.cache),typeof(dv)}(
        sol, old.u, old.k, old.t, old.dt, old.uprev, old.tprev, old.alg,
        old.dtcache, old.tdir, old.saveiter, old.saveiter_dense, old.cache, dv)
    DelayDiffEq.HistoryFunction(prob.h, saved)
end

function _reserve_history!(integ)
    records = integ.integrator.sol.k
    records isa HistoryBuffer || return nothing
    for i in (length(records)+1):length(records.slots)
        _copy_record!(records.slots[_slot(records, i)], integ.k)
    end
    nothing
end

mutable struct RealTimeAudioDEData
    problem::AbstractSciMLProblem
    alg::AbstractDEAlgorithm
    control::RealTimeAudioDEControlData
    state::RealTimeAudioDEStateData
    stream_data::RealTimeAudioDEStreamData
    rb::Union{RBAudio,Nothing}
    producer::Union{Task,Nothing}
    integrator::Any
    renderer::Any
    buffer::Vector{Float32}
    running::Atomic{Bool}
    frames_cb::Atomic{Int}
    samples_popped::Atomic{Int}
    underflows::Atomic{Int}
    framesize_mismatch::Atomic{Int}
    pushed_samples::Atomic{Int}
    solver_failures::Atomic{Int}
    delay_window::Union{Nothing,DelayWindow}
end

mutable struct DESource
    data::RealTimeAudioDEData
    callback::Base.CFunction
end

# PortAudio keeps a raw userdata pointer, not a Julia GC root.
const _active_sources = IdDict{DESource,Nothing}()
const _active_sources_lock = ReentrantLock()

function normalize_channel_map(channel_map, nvars::Int, nch::Int=
        channel_map isa AbstractMatrix ? size(channel_map, 2) : length(channel_map))
    nch > 0 || throw(ArgumentError("channel_map needs at least one channel"))
    G = zeros(Float64, nvars, nch)
    if channel_map isa AbstractVector{<:Integer}
        for ch in 1:min(nch, length(channel_map))
            k = channel_map[ch]
            0 <= k <= nvars || throw(ArgumentError("variable $k is out of bounds"))
            k == 0 || (G[k, ch] = 1.0)
        end
    elseif channel_map isa AbstractVector{<:AbstractVector{<:Integer}}
        for ch in 1:min(nch, length(channel_map)), k in channel_map[ch]
            0 <= k <= nvars || throw(ArgumentError("variable $k is out of bounds"))
            k == 0 || (G[k, ch] += 1.0)
        end
    elseif channel_map isa AbstractMatrix{<:Real}
        size(channel_map, 1) <= nvars || throw(DimensionMismatch("too many map rows"))
        nch <= size(channel_map, 2) || throw(DimensionMismatch("too few map columns"))
        for ch in 1:nch, k in axes(channel_map, 1)
            G[k, ch] = channel_map[k, ch]
        end
        all(isfinite, G) || throw(ArgumentError("channel gains must be finite"))
    else
        throw(ArgumentError("use variable indices, lists of indices, or a gain matrix"))
    end
    G
end

"""
    DESource(f, u0::Vector{Float64}, p::Vector{Float64}; alg=Tsit5(), channel_map=[1, 1])
    DESource(f, g, u0::Vector{Float64}, p::Vector{Float64}; alg=SOSRA(), channel_map=[1, 1])

Create an ODE or SDE audio source, copying its inputs. Maps accept variable
indices (0 means silence), lists of indices, or a variables-by-channels matrix.
"""
function DESource(f, u0::Vector{Float64}, p::Vector{Float64};
        alg=Tsit5(), channel_map=[1, 1])
    _DESource(ODEProblem(f, copy(u0), (0.0, 0.01), copy(p)), alg, channel_map)
end

function DESource(f, g, u0::Vector{Float64}, p::Vector{Float64};
        alg=SOSRA(), channel_map=[1, 1])
    _DESource(SDEProblem(f, g, copy(u0), (0.0, 0.01), copy(p)), alg, channel_map)
end

function DESource(f::SDEFunction, u0::Vector{Float64}, p::Vector{Float64};
        alg=SOSRA(), channel_map=[1, 1])
    _DESource(SDEProblem(f, copy(u0), (0.0, 0.01), copy(p)), alg, channel_map)
end

function _DESource(prob::AbstractSciMLProblem, alg::AbstractDEAlgorithm, channel_map; delay_window=nothing)
    isempty(prob.u0) && throw(ArgumentError("state must not be empty"))
    all(isfinite, prob.u0) || throw(ArgumentError("initial state must be finite"))
    all(isfinite, prob.p) || throw(ArgumentError("parameters must be finite"))
    G = normalize_channel_map(channel_map, length(prob.u0))
    c = RealTimeAudioDEControlData(copy(prob.u0), copy(prob.p), 1.0, 1.0, G,
        nothing, ReentrantLock())
    state = RealTimeAudioDEStateData(Float64(first(prob.tspan)), copy(prob.u0))
    stream = RealTimeAudioDEStreamData(-1.0, 0, 0, Ref{Ptr{PaStream}}(C_NULL), false)
    d = RealTimeAudioDEData(prob, alg, c, state, stream, nothing, nothing, nothing,
        nothing, Float32[], Atomic{Bool}(false), Atomic{Int}(0), Atomic{Int}(0),
        Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0), delay_window)
    callback = create_callback()
    callback_ptr = @cfunction($callback, PaStreamCallbackResult,
        (Ptr{Cvoid}, Ptr{Cvoid}, Culong, Ptr{PaStreamCallbackTimeInfo},
         PaStreamCallbackFlags, Ptr{Cvoid}))
    DESource(d, callback_ptr)
end


"""
    DESource(problem::DDEProblem; alg=MethodOfSteps(Tsit5()), max_delay,
             channel_map=[1, 1], abstol=1e-12, reltol=1e-10,
             history_capacity=100_000, dtmax=nothing, reuse_history=true)

Use DelayDiffEq's solver and interpolation with a rolling history window.
Supports in-place Float64 equations with positive constant lags; neutral
equations currently require one lag. History records are preallocated and reused;
history_capacity limits retained interpolation knots; exhaustion silences output.
Set reuse_history=false to use the original allocating history storage.
With dtmax, sample the native interpolant between adaptive steps and stop at block
boundaries. Without it, limit steps to the audio substep grid.
"""
function DESource(prob::DDEProblem; alg=MethodOfSteps(Tsit5()),
        max_delay::Real=maximum(prob.constant_lags; init=0.0), channel_map=[1, 1],
        abstol::Real=1e-12, reltol::Real=1e-10, history_capacity::Integer=100_000,
        dtmax::Union{Nothing,Real}=nothing, reuse_history::Bool=true)
    SciMLBase.isinplace(prob) || throw(ArgumentError("DDE dynamics must be in-place"))
    prob.u0 isa Vector{Float64} && prob.p isa Vector{Float64} ||
        throw(ArgumentError("DDE state and parameters must be Float64 vectors"))
    isempty(prob.dependent_lags) || throw(ArgumentError("declare constant delays"))
    isempty(prob.kwargs) || throw(ArgumentError("DDE callbacks and solver keywords are not supported"))
    isone(prob.f.mass_matrix) || throw(ArgumentError("DDE mass matrix must be the identity"))
    alg isa MethodOfSteps || throw(ArgumentError("pass a DelayDiffEq MethodOfSteps algorithm"))
    lags = Tuple(Float64.(prob.constant_lags))
    !isempty(lags) && all(x -> isfinite(x) && x > 0, lags) ||
        throw(ArgumentError("declare finite, positive constant_lags"))
    prob.neutral && length(lags) != 1 &&
        throw(ArgumentError("neutral audio currently supports one constant lag"))
    isfinite(max_delay) && max_delay >= maximum(lags) ||
        throw(ArgumentError("max_delay must cover all declared lags"))
    isfinite(first(prob.tspan)) || throw(ArgumentError("initial time must be finite"))
    all(x -> isfinite(x) && x > 0, (abstol, reltol)) ||
        throw(ArgumentError("DDE tolerances must be finite and positive"))
    history_capacity >= 4 || throw(ArgumentError("history_capacity must be at least four"))
    dtmax === nothing || (isfinite(dtmax) && 0 < dtmax <= minimum(lags)) ||
        throw(ArgumentError("dtmax must be positive and no greater than the smallest lag"))
    owned = remake(prob; u0=copy(prob.u0), p=copy(prob.p), constant_lags=lags)
    window = DelayWindow(Float64(max_delay), Float64(first(prob.tspan)), NaN, 0,
        Float64(abstol), Float64(reltol), Int(history_capacity), dtmax === nothing ? 0.0 : Float64(dtmax), reuse_history)
    _DESource(owned, alg, channel_map; delay_window=window)
end

_history(::Nothing, integ) = nothing
_history(::DelayWindow, integ) = integ.history
_trim_history!(::Nothing, integ) = nothing

# DelayDiffEq 6.3.1 storage bookkeeping; interpolation remains entirely in SciML.
function _trim_history!(h::DelayWindow, integ)
    saved = integ.integrator
    sol = saved.sol
    remove = min(searchsortedfirst(sol.t, min(integ.t, h.start + h.steps*h.dt) - h.max_delay) - 3,
        integ.prev_idx - 1, integ.prev2_idx - 1)
    if remove > 0
        deleteat!(sol.t, 1:remove)
        deleteat!(sol.u, 1:remove)
        deleteat!(sol.k, 1:remove)
        sol.alg_choice === nothing || deleteat!(sol.alg_choice, 1:remove)
        saved.saveiter -= remove
        saved.saveiter_dense -= remove
        integ.prev_idx -= remove
        integ.prev2_idx -= remove
        sol.interp.ts_hint.idx_prev = max(0, sol.interp.ts_hint.idx_prev - remove)
        OrdinaryDiffEqCore.reprobe_ts_hint!(sol.interp.ts_hint)
    end
    # Only dependent delays use this log; future constant-lag events stay queued.
    empty!(integ.tracked_discontinuities)
    nothing
end

function _advance!(::Nothing, integ)
    previous = integ.t
    step!(integ)
    integ.t > previous && all(isfinite, integ.u) && successful_retcode(check_error(integ))
end

function _advance!(h::DelayWindow, integ)
    target = h.start + (h.steps + 1) * h.dt
    h.dtmax == 0 && target <= integ.t && return false
    h.dtmax == 0 && add_tstop!(integ, target)
    while integ.t < target
        if length(integ.integrator.sol.t) >= h.capacity
            _trim_history!(h, integ)
            length(integ.integrator.sol.t) >= h.capacity && return false
        end
        previous = integ.t
        step!(integ)
        integ.t > previous && all(isfinite, integ.u) &&
            successful_retcode(check_error(integ)) || return false
    end
    h.steps += 1
    h.dtmax > 0 || abs(integ.t - target) <= 16eps(max(abs(target), 1.0))
end

function _prepare_integrator(d, prob, p, dt, reset)
    prepared = remake(prob; u0=copy(d.state.u), p=copy(p), tspan=(d.state.t, Inf))
    integ = init(prepared, d.alg; dt, adaptive=false, save_on=false,
        save_start=false, save_end=false, save_everystep=false, dense=false,
        save_noise=false, maxiters=typemax(Int))
    integ, nothing
end

function _prepare_integrator(d, prob::DDEProblem, p, dt, reset)
    dt <= minimum(prob.constant_lags) || throw(ArgumentError("step must not exceed the smallest lag"))
    if d.integrator !== nothing && reset === nothing
        old = d.integrator
        # Retain the complete solver, including its history, across stop/restart.
        old.p == p || throw(ArgumentError("reset_state! before changing DDE parameters"))
        h = d.delay_window
        dt == h.dt || throw(ArgumentError("reset_state! before changing the DDE timestep"))
        return old, h
    end
    prepared = remake(prob; u0=copy(d.state.u), p=copy(p), tspan=(d.state.t, Inf))
    h = d.delay_window
    h.reuse && (prepared = remake(prepared; h=ReuseHistory(prob.h, h.capacity)))
    maximum_step = h.dtmax > 0 ? h.dtmax : dt
    integ = init(prepared, d.alg; dt, dtmax=maximum_step, dtmin=0.0, adaptive=true, abstol=h.abstol,
        reltol=h.reltol, save_on=false, save_start=false, save_end=false,
        save_everystep=false, dense=false, maxiters=typemax(Int))
    _reserve_history!(integ)
    h.start, h.dt, h.steps = d.state.t, dt, 0
    integ, h
end

# SciMLBase 3 renamed this public operation; retain the 2.116 API as well.
@static if isdefined(SciMLBase, :derivative_discontinuity!)
    _parameters_changed!(integ) = SciMLBase.derivative_discontinuity!(integ, true)
else
    _parameters_changed!(integ) = SciMLBase.u_modified!(integ, true)
end

function _apply_controls!(r::AudioRenderer, c)
    integ = r.integrator
    p = @atomic c.p
    if r.history !== nothing && (@atomic c.ts) != r.timescale
        throw(ArgumentError("stop, reset_state!, and prepare! to change the DDE timestep"))
    end
    if r.history !== nothing && (@atomic c.reset) === nothing &&
            p != integ.p
        throw(ArgumentError("reset_state! before changing DDE parameters or timestep"))
    end
    if p !== r.parameters
        copyto!(integ.p, p)
        _parameters_changed!(integ)
        r.parameters = p
    end
    ts = @atomic c.ts
    if ts != r.timescale
        set_proposed_dt!(integ, ts / r.sample_rate / r.substeps)
        r.timescale = ts
    end
    reset = @atomic c.reset
    if reset !== nothing
        reset = @atomicswap c.reset = nothing
        if reset !== nothing
            t0 = Float64(first(integ.sol.prob.tspan))
            reinit!(integ, reset; t0, tf=Inf, reset_dt=false)
            if r.history !== nothing
                r.history.start = t0
                r.history.steps = 0
                r.history.dt = r.timescale / r.sample_rate / r.substeps
            end
            r.failed = false
        end
    end
    nothing
end

function _render_block!(out::AbstractVector{Float32}, r::AudioRenderer, c, nch::Int, observe=nothing)
    _apply_controls!(r, c)
    if r.failed
        fill!(out, 0.0f0)
        return nothing
    end
    G = @atomic c.channel_map
    gain = @atomic c.gain
    integ = r.integrator
    nvars = length(integ.u)
    index = firstindex(out)
    h = r.history
    interpolated = h !== nothing && h.dtmax > 0
    if interpolated && !isempty(out)
        add_tstop!(integ, h.start + (h.steps + length(out) ÷ nch*r.substeps)*h.dt)
    end
    # Sample before stepping, matching upstream's save_start=true.
    for _ in 1:(length(out) ÷ nch)
        sample_time = interpolated ? h.start + h.steps*h.dt : integ.t
        state = if interpolated
            integ.history(r.sample, integ.p, sample_time)
            r.sample
        else
            integ.u
        end
        observed = observe === nothing ? nothing : observe(state, integ.p, sample_time, _history(h, integ))
        @inbounds for ch in 1:nch
            sample = 0.0
            if observed === nothing
                @simd for k in 1:nvars
                    sample += G[k, ch] * state[k]
                end
            else
                sample = observed[ch]
            end
            value = Float32(gain * sample)
            if !isfinite(value)
                r.failed = true
                fill!(out, 0.0f0)
                return nothing
            end
            out[index] = value
            index += 1
        end
        for _ in 1:r.substeps
            if !_advance!(r.history, integ)
                r.failed = true
                fill!(out, 0.0f0)
                return nothing
            end
        end
    end
    _trim_history!(r.history, integ)
    nothing
end

"""
    prepare!(source; sample_rate=48000.0, buffer_size=256, substeps=1)

Prepare a stopped source without opening a device. ODE/SDEs take `substeps`
fixed steps per sample. DDEs use adaptive steps. An explicit dtmax samples their native interpolant;
otherwise steps end on the audio substep grid.
"""
function prepare!(source::DESource; sample_rate::Real=48000.0,
        buffer_size::Integer=256, substeps::Integer=1, n_channels::Integer=size(
            @atomic(source.data.control.channel_map), 2))
    d = source.data
    d.running[] && throw(ArgumentError("stop the source before preparing it"))
    sample_rate = Float64(sample_rate)
    isfinite(sample_rate) && sample_rate > 0 ||
        throw(ArgumentError("sample_rate must be finite and positive"))
    buffer_size > 0 || throw(ArgumentError("buffer_size must be positive"))
    substeps > 0 || throw(ArgumentError("substeps must be positive"))
    G = @atomic d.control.channel_map
    1 <= n_channels <= size(G, 2) || throw(ArgumentError("invalid channel count"))
    _save_state!(d)
    reset = @atomicswap d.control.reset = nothing
    if reset !== nothing
        d.state.t = Float64(first(d.problem.tspan))
        copyto!(d.state.u, reset)
    end
    p = @atomic d.control.p
    ts = @atomic d.control.ts
    integ, history = _prepare_integrator(d, d.problem, p, ts / sample_rate / substeps, reset)
    d.delay_window = history
    d.integrator = integ
    d.renderer = AudioRenderer(integ, history, p, sample_rate, ts, Int(substeps), false, copy(integ.u))
    d.stream_data.sample_rate = sample_rate
    d.stream_data.buffer_size = buffer_size
    d.stream_data.n_channels = n_channels
    d.buffer = zeros(Float32, buffer_size * n_channels)
    source
end

"""
    render!(out::AbstractVector{Float32}, source; observe=nothing)

Fill caller-owned interleaved audio using a prepared, stopped source. Controls
apply once per block. Solver failure silences output until reset or preparation.
Optional observe(u,p,t,h) returns channel values in place of the channel map;
h is SciML's history function for DDEs and nothing for ODE/SDEs.
"""
function render!(out::AbstractVector{Float32}, source::DESource; observe=nothing)
    d = source.data
    d.running[] && throw(ArgumentError("offline rendering requires a stopped source"))
    d.renderer === nothing && throw(ArgumentError("call prepare! before render!"))
    nch = Int(d.stream_data.n_channels)
    length(out) % nch == 0 || throw(DimensionMismatch("output must contain whole frames"))
    _render_block!(out, d.renderer, d.control, nch, observe)
    out
end

function _save_state!(d)
    d.renderer === nothing && return nothing
    _save_state!(d.state, d.renderer)
end

function _save_state!(state, r::AudioRenderer)
    state.t = r.integrator.t
    copyto!(state.u, r.integrator.u)
    nothing
end

function RBAudio(n::Integer)
    n > 0 || throw(ArgumentError("ring capacity must be positive"))
    capacity = nextpow(2, max(Int(n), 1024))
    RBAudio(zeros(Float32, capacity), capacity - 1, Atomic{Int}(0), Atomic{Int}(0))
end

@inline rb_capacity(rb::RBAudio) = length(rb.buf)
@inline rb_available_to_read(rb::RBAudio) = rb.w[] - rb.r[]
@inline rb_available_to_write(rb::RBAudio) = rb_capacity(rb) - rb_available_to_read(rb)
@inline rb_is_empty(rb::RBAudio) = rb_available_to_read(rb) == 0
@inline rb_is_full(rb::RBAudio) = rb_available_to_write(rb) == 0

function rb_write!(rb::RBAudio, samples::Vector{Float32})
    n = length(samples)
    w = rb.w[]
    n <= rb_capacity(rb) - (w - rb.r[]) || return false
    wi = w & rb.mask
    tail = min(n, rb_capacity(rb) - wi)
    copyto!(rb.buf, wi + 1, samples, 1, tail)
    copyto!(rb.buf, 1, samples, tail + 1, n - tail)
    rb.w[] = w + n
    true
end

# n and each published block must contain whole frames.
function rb_pop!(rb::RBAudio, out::Ptr{Cfloat}, n::Int)
    r = rb.r[]
    m = min(rb.w[] - r, n)
    m <= 0 && return 0
    ri = r & rb.mask
    tail = min(m, rb_capacity(rb) - ri)
    storage = rb.buf
    GC.@preserve storage begin
        unsafe_copyto!(out, pointer(storage, ri + 1), tail)
        if tail < m
            # Pointer offsets are in bytes.
            unsafe_copyto!(out + tail * sizeof(Cfloat), pointer(storage), m - tail)
        end
    end
    rb.r[] = r + m
    m
end

# Callback work is limited to copying samples, silence, and counters.
function _audio_callback!(data::RealTimeAudioDEData, out::Ptr{Cfloat}, frames::Int)
    nch = Int(data.stream_data.n_channels)
    nsamp = frames * nch
    data.frames_cb[] += 1
    expected = Int(data.stream_data.buffer_size)
    if expected != 0 && frames != expected
        data.framesize_mismatch[] += 1
    end
    rb = data.rb
    popped = rb === nothing ? 0 : rb_pop!(rb, out, nsamp)
    data.samples_popped[] += popped
    if popped < nsamp
        data.underflows[] += 1
        for i in (popped + 1):nsamp
            unsafe_store!(out, 0.0f0, i)
        end
    end
    paContinue
end

function create_callback()
    function callback(input::Ptr{Cvoid}, output::Ptr{Cvoid}, frames::Culong,
            time_info::Ptr{PaStreamCallbackTimeInfo}, flags::PaStreamCallbackFlags,
            userdata::Ptr{Cvoid})::PaStreamCallbackResult
        data = unsafe_pointer_to_objref(userdata)::RealTimeAudioDEData
        _audio_callback!(data, Ptr{Cfloat}(output), Int(frames))
    end
    callback
end

function _produce_block!(d, renderer::AudioRenderer, rb::RBAudio)
    was_failed = renderer.failed
    _render_block!(d.buffer, renderer, d.control, Int(d.stream_data.n_channels))
    if renderer.failed && !was_failed
        d.solver_failures[] += 1
    end
    rb_write!(rb, d.buffer) || error("producer attempted to overwrite unread audio")
    d.pushed_samples[] += length(d.buffer)
    nothing
end

function _produce_audio!(d, renderer::AudioRenderer, rb::RBAudio, target, timer, ready)
    notify(ready)
    try
        while d.running[]
            while d.running[] && rb_available_to_read(rb) + length(d.buffer) <= target
                _produce_block!(d, renderer, rb)
            end
            wait(timer)
        end
    catch err
        if d.running[]
            d.solver_failures[] += 1
            @error "Audio producer stopped" exception=(err, catch_backtrace())
        end
    finally
        close(timer)
        d.running[] = false
    end
    nothing
end

"""
    start_DESource(source, output_device; sample_rate=-1.0, buffer_size=0,
                   substeps=1, queue_blocks=4)

Prepare, prefill, and start playback. Requires Julia with at least two worker
threads. Default block size: 256 frames. More queued blocks add control latency.
"""
function start_DESource(source::DESource, output_device::PaDeviceIndex;
        sample_rate::Real=-1.0, buffer_size::Integer=0,
        substeps::Integer=1, queue_blocks::Integer=4)
    d = source.data
    d.running[] && return nothing
    d.stream_data.stream[] != C_NULL && stop_DESource(source)
    Base.Threads.nthreads() >= 2 ||
        throw(ArgumentError("live audio requires Julia started with --threads=2 or more"))
    queue_blocks >= 2 || throw(ArgumentError("queue_blocks must be at least 2"))
    buffer_size >= 0 || throw(ArgumentError("buffer_size must be nonnegative"))
    info = get_device_info(output_device)
    info.maxOutputChannels > 0 || throw(ArgumentError("device has no output channels"))
    fs = sample_rate < 0 ? info.defaultSampleRate : Float64(sample_rate)
    frames = buffer_size == 0 ? 256 : Int(buffer_size)
    map_channels = size(@atomic(d.control.channel_map), 2)
    nch = min(map_channels, Int(info.maxOutputChannels))
    nch < map_channels && @warn "output device has fewer channels than channel map"
    prepare!(source; sample_rate=fs, buffer_size=frames, substeps, n_channels=nch)
    target = queue_blocks * length(d.buffer)
    rb = RBAudio(target)
    d.rb = rb
    for counter in (d.frames_cb, d.samples_popped, d.underflows, d.framesize_mismatch,
            d.pushed_samples, d.solver_failures)
        counter[] = 0
    end
    # Prefill with the first samples while compiling the audio path.
    for _ in 1:queue_blocks
        _produce_block!(d, d.renderer, rb)
    end
    _audio_callback!(d, Ptr{Cfloat}(C_NULL), 0)
    d.frames_cb[] = d.framesize_mismatch[] = 0
    params = PaStreamParameters(output_device, nch, paFloat32,
        info.defaultLowOutputLatency, C_NULL)
    lock(_active_sources_lock) do
        _active_sources[source] = nothing
    end
    try
        err = Pa_OpenStream(d.stream_data.stream, C_NULL, Ref(params), fs,
            Culong(frames), 0, source.callback, pointer_from_objref(d))
        err == 0 || error(unsafe_string(Pa_GetErrorText(err)))
        # Reuse one timer instead of allocating on every poll.
        timer = Timer(0; interval=max(0.001, frames / fs / 4))
        ready = Base.Event()
        d.running[] = true
        d.producer = Base.Threads.@spawn _produce_audio!(d, d.renderer, rb, target, timer, ready)
        wait(ready)
        err = Pa_StartStream(d.stream_data.stream[])
        err == 0 || error(unsafe_string(Pa_GetErrorText(err)))
        d.stream_data.ever_started = true
    catch
        stop_DESource(source)
        rethrow()
    end
    nothing
end

"""Stop the producer, close PortAudio, and preserve the last rendered solver state."""
function stop_DESource(source::DESource)
    d = source.data
    d.running[] = false
    if d.producer !== nothing
        wait(d.producer)
        d.producer = nothing
    end
    stream = d.stream_data.stream[]
    if stream != C_NULL
        # CloseStream waits for the callback before its storage is released.
        err = Pa_CloseStream(stream)
        err == 0 || error(unsafe_string(Pa_GetErrorText(err)))
        d.stream_data.stream[] = C_NULL
    end
    _save_state!(d)
    d.rb = nothing
    lock(_active_sources_lock) do
        delete!(_active_sources, source)
    end
    nothing
end

isinitialized(source::DESource) = source.data.stream_data.ever_started
isactive(source::DESource) = source.data.stream_data.stream[] != C_NULL &&
    Pa_IsStreamActive(source.data.stream_data.stream[]) == 1
isstopped(source::DESource) = isinitialized(source) && !isactive(source)

"""Set initial conditions used by the next `reset_state!`; does not reset by itself."""
function set_u0!(source::DESource, u::Vector{Float64})
    length(u) == length(source.data.state.u) || throw(DimensionMismatch("u0 length changed"))
    all(isfinite, u) || throw(ArgumentError("initial state must be finite"))
    @atomic source.data.control.u0 = copy(u)
    nothing
end
get_u0(source::DESource) = copy(@atomic source.data.control.u0)

"""Request a reset to the current `u0` at the next rendered block boundary."""
function reset_state!(source::DESource)
    @atomic source.data.control.reset = copy(@atomic source.data.control.u0)
    nothing
end

function set_param!(source::DESource, index::Int, value::Real)
    value = Float64(value)
    isfinite(value) || throw(ArgumentError("parameter must be finite"))
    c = source.data.control
    lock(c.writer_lock) do
        p = copy(@atomic c.p)
        p[index] = value
        @atomic c.p = p
    end
    nothing
end
get_param(source::DESource, index::Int) = (@atomic source.data.control.p)[index]
getparam(source::DESource, index::Int) = get_param(source, index)
get_params(source::DESource) = copy(@atomic source.data.control.p)

function set_ts!(source::DESource, value::Real)
    value = Float64(value)
    isfinite(value) && value > 0 || throw(ArgumentError("time scale must be finite and positive"))
    @atomic source.data.control.ts = value
    nothing
end
get_ts(source::DESource) = @atomic source.data.control.ts

function set_gain!(source::DESource, value::Real)
    value = Float64(value)
    isfinite(value) || throw(ArgumentError("gain must be finite"))
    @atomic source.data.control.gain = value
    nothing
end
get_gain(source::DESource) = @atomic source.data.control.gain

function set_channelmap!(source::DESource, channel_map)
    d = source.data
    G = normalize_channel_map(channel_map, length(d.state.u))
    if d.renderer !== nothing && size(G, 2) != size(@atomic(d.control.channel_map), 2)
        d.running[] && throw(ArgumentError("stop the source before changing channel count"))
        _save_state!(d)
        d.renderer = nothing
        d.integrator = nothing
    end
    @atomic d.control.channel_map = G
    nothing
end
get_channelmap(source::DESource) = copy(@atomic source.data.control.channel_map)

function audio_debug_status(source::DESource)
    d = source.data
    rb = d.rb
    (; frames_cb=d.frames_cb[], samples_popped=d.samples_popped[],
       pushed_samples=d.pushed_samples[], underflows=d.underflows[],
       framesize_mismatch=d.framesize_mismatch[], solver_failures=d.solver_failures[],
       rb_level=rb === nothing ? -1 : rb_available_to_read(rb),
       rb_capacity=rb === nothing ? -1 : rb_capacity(rb))
end

function _stopped_integrator(source)
    source.data.running[] && throw(ArgumentError("stop the producer before inspecting its integrator"))
    source.data.integrator === nothing && throw(ArgumentError("call prepare! first"))
    source.data.integrator
end

function integrator_info(source::DESource)
    integ = _stopped_integrator(source)
    (; t=integ.t, dt=integ.dt, eltype_u=eltype(integ.u), eltype_p=eltype(integ.p))
end

function current_mix_snapshot(source::DESource)
    integ = _stopped_integrator(source)
    G = @atomic source.data.control.channel_map
    sample = zeros(size(G, 2))
    for ch in axes(G, 2), k in axes(G, 1)
        sample[ch] += G[k, ch] * integ.u[k]
    end
    sample .*= get_gain(source)
    (; t=integ.t, sample)
end

function probe_integrator!(source::DESource; n=1024, take=16)
    integ = _stopped_integrator(source)
    n >= 0 && take >= 0 || throw(ArgumentError("n and take must be nonnegative"))
    vals = Vector{Float32}(undef, min(n, take))
    changed = false
    last = Float32(integ.u[1])
    for i in 1:n
        step!(integ)
        value = Float32(integ.u[1])
        i <= length(vals) && (vals[i] = value)
        changed |= value != last
        last = value
    end
    (; changed, first_vals=vals)
end

"""Get the default audio output device index."""
function get_default_output_device()::PaDeviceIndex
    return Pa_GetDefaultOutputDevice()
end

"""Report information about an audio device."""
function get_device_info(device::PaDeviceIndex)::PaDeviceInfo
    if Pa_GetDeviceInfo(device) == C_NULL
        error("invalid device index.")
    end
    return unsafe_load(Pa_GetDeviceInfo(device))
end

"""Return an array of available audio devices."""
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

"""Print a list of available audio devices."""
function list_devices()
    for i in 0:Pa_GetDeviceCount() - 1
        device_info = unsafe_load(Pa_GetDeviceInfo(i))
        println("Device: $i, ", device_info)
    end
end

"""Get the index of an audio device by its name."""
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
    print(io, "DESource(", typeof(source.data.alg), ", ",
        length(source.data.state.u), " variables, ",
        size(@atomic(source.data.control.channel_map), 2), " channels)")
end
