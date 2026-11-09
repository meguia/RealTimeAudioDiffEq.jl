using RealTimeAudioDiffEq
using SciMLBase: DDEProblem

cubic_lag(u, p, t) = p[1] + p[2]*sin(p[3]*t)

# x′(t) = x(t − τ) * (1 − x(t − τ)^2).
function cubic_delay!(du, u, h, p, t)
    z = h(p, t-cubic_lag(u,p,t); idxs=1)
    du[1] = z*(1-z*z)
    nothing
end

cubic_history(p, t; idxs=nothing) = idxs === nothing ? [0.5] : 0.5

function cubic_source(; timescale=400.0)
    # One smooth delay sweep every five audio seconds.
    p = [1.0, 0.4, 2pi/(5timescale)]
    prob = DDEProblem(cubic_delay!, [0.52], cubic_history, (0.0, Inf), p;
        dependent_lags=(cubic_lag,))
    source = DESource(prob; min_delay=0.5, max_delay=2.0,
        dtmax=0.0625, history_capacity=8192,
        abstol=1e-10, reltol=1e-8, channel_map=[1, 1])
    set_ts!(source, timescale)
    set_gain!(source, 0.15)
    source
end

function cubic_offline(source)
    prepare!(source; sample_rate=48000, buffer_size=256)
    buffer = zeros(Float32, 512)
    for _ in 1:1000
        render!(buffer, source)
        source.data.renderer.failed && error("solver stopped")
        all(isfinite, buffer) || error("nonfinite output")
    end
    println("Offline delay sweep passed; no audio device opened.")
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
        println("τ sweeps smoothly between 0.6 and 1.4. Press Enter to stop.")
        readline()
    finally
        stop_DESource(source)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && cubic_main()
