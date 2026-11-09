using RealTimeAudioDiffEq
using SciMLBase: DDEProblem

# x′(t) = x(t − τ) * (1 − x(t − τ)^2).
function cubic_delay!(du, u, h, p, t)
    z = h(p, t - p[1]; idxs=1)
    du[1] = z * (1 - z*z)
    nothing
end

cubic_history(p, t; idxs=nothing) = idxs === nothing ? [0.5] : 0.5

function cubic_source(; tau=1.0, timescale=400.0)
    prob = DDEProblem(cubic_delay!, [0.52], cubic_history, (0.0, Inf), [Float64(tau)];
        constant_lags=(Float64(tau),))
    source = DESource(prob; delay_parameter=1, min_delay=0.5, max_delay=2.0,
        delay_smoothing=0.05, dtmax=0.0625, history_capacity=8192,
        abstol=1e-10, reltol=1e-8, channel_map=[1, 1])
    set_ts!(source, timescale)
    set_gain!(source, 0.15)
    source
end

function cubic_offline(source)
    prepare!(source; sample_rate=48000, buffer_size=256)
    buffer = zeros(Float32, 512)
    for tau in (1.0, 1.2, 0.6, 1.4)
        set_param!(source, 1, tau)
        for _ in 1:50
            render!(buffer, source)
            source.data.renderer.failed && error("solver stopped at τ = $tau")
            all(isfinite, buffer) || error("nonfinite output at τ = $tau")
        end
        println("τ target = $tau; model time = ", integrator_info(source).t)
    end
    println("Offline check passed; no audio device opened.")
end

function cubic_main(args=ARGS)
    if args == ["--devices"]
        foreach(println, get_devices())
        return
    end
    source = cubic_source()
    args == ["--offline"] && return cubic_offline(source)
    device = isempty(args) ? get_default_output_device() : Cint(parse(Int, only(args)))
    start_DESource(source, device; sample_rate=48000, buffer_size=256)
    try
        println("Enter τ in [0.5, 2.0]; q stops playback.")
        while !eof(stdin)
            command = strip(readline())
            command == "q" && break
            tau = tryparse(Float64, command)
            if tau === nothing || !isfinite(tau) || !(0.5 <= tau <= 2.0)
                println("Enter a delay between 0.5 and 2.0, or q.")
                continue
            end
            set_param!(source, 1, tau)
            println("τ target = $tau")
        end
    finally
        stop_DESource(source)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && cubic_main()
