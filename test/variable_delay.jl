using DelayDiffEq: MethodOfSteps
using SciMLBase: DDEProblem, init, step!

vd_constant_history(p, t; idxs=nothing) = idxs === nothing ? [0.5] : 0.5
vd_sine_history(p, t; idxs=nothing) = idxs === nothing ? [sin(8t)] : sin(8t)
vd_sine_history(p, t, ::Type{Val{1}}; idxs=nothing) =
    idxs === nothing ? [8cos(8t)] : 8cos(8t)

function vd_cubic!(du, u, h, p, t)
    z = h(p, t-p[1]; idxs=1)
    du[1] = z*(1-z*z)
    nothing
end

function vd_sine!(du, u, h, p, t)
    du[1] = 8cos(8t) - (u[1]-sin(8t)) +
        0.2*(h(p, t-p[1]; idxs=1)-sin(8(t-p[1])))
    nothing
end

function vd_neutral!(du, u, h, p, t)
    du[1] = 8cos(8t) - (u[1]-sin(8t)) +
        0.2*(h(p, t-p[1], Val{1}; idxs=1)-8cos(8(t-p[1])))
    nothing
end

const VD_SCHEDULE = ((1.0, 0.2, 0.6, 1.2), (4.0, 0.6, 0.1, 1.5),
    (8.0, 0.1, 0.7, 1.8), (12.0, 0.7, 0.2, 1.5))

function vd_reference_lag(u, p, t)
    value = 0.2
    for (origin, before, after, duration) in VD_SCHEDULE
        t < origin && return value
        if t < origin+duration
            x = (t-origin)/duration
            return before + (after-before)*(3x^2-2x^3)
        end
        value = after
    end
    value
end

struct VDReference{F}
    f::F
    parameters::Vector{Float64}
end

function (f::VDReference)(du, u, h, p, t)
    f.parameters[1] = vd_reference_lag(u, p, t)
    f.f(du, u, h, f.parameters, t)
end

function vd_advance_reference!(integ, target)
    RT.SciMLBase.add_tstop!(integ, target)
    while integ.t < target
        previous = integ.t
        step!(integ)
        integ.t > previous || error("reference solver did not advance")
    end
end

@testset "Bounded variable-delay validation and ownership" begin
    p = [0.5]
    prob = DDEProblem(vd_cubic!, [0.52], vd_constant_history, (0.0, 20.0), p)
    options = (delay_parameter=1, min_delay=0.1, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    source = DESource(prob; options...)
    @test isempty(source.data.problem.constant_lags)
    @test length(source.data.problem.dependent_lags) == 1
    p[1] = 0.6
    @test get_param(source, 1) == 0.5
    for bad in ((min_delay=0.0,), (min_delay=-0.1,), (min_delay=NaN,),
            (max_delay=Inf,), (max_delay=0.4,), (min_delay=0.6,),
            (min_delay=nothing,), (max_delay=nothing,), (dtmax=0.11,),
            (delay_smoothing=0.0,), (delay_smoothing=NaN,),
            (discontinuity_capacity=3,), (delay_parameter=0,), (delay_parameter=2,))
        @test_throws ArgumentError DESource(remake(prob; p=[0.5]); merge(options, bad)...)
    end
    @test_throws ArgumentError DESource(remake(prob; constant_lags=(0.4,)); options...)
    @test_throws ArgumentError DESource(remake(prob; constant_lags=(0.5, 0.6)); options...)
    @test_throws ArgumentError DESource(remake(prob; dependent_lags=((u,p,t)->p[1],)); options...)
    @test_throws ArgumentError DESource(remake(prob; p=[0.5]); options..., alg=MethodOfSteps(Rodas5P()))
    for value in (0.0, 0.099, 0.701, Inf, NaN)
        @test_throws ArgumentError set_param!(source, 1, value)
        @test get_param(source, 1) == 0.5
    end
    @test_throws ArgumentError prepare!(source; sample_rate=5)
    for value in (0.1, 0.7)
        set_param!(source, 1, value)
        @test get_param(source, 1) == value
    end
end

@testset "Moving cubic and neutral delays match unbounded native integration" begin
    for neutral in (false, true)
        f = neutral ? vd_neutral! : vd_cubic!
        history = neutral ? vd_sine_history : vd_constant_history
        initial = neutral ? [0.0] : [0.52]
        prob = DDEProblem(f, initial, history, (0.0, 20.0), [0.2];
            constant_lags=(0.2,), neutral)
        source = DESource(prob; delay_parameter=1, min_delay=0.1, max_delay=0.7,
            delay_smoothing=0.05, dtmax=0.01, history_capacity=256, channel_map=[1])
        prepare!(source; sample_rate=200, buffer_size=20)
        reference = DDEProblem(VDReference(f, [0.2]), copy(initial), history,
            (0.0, Inf), [0.2]; dependent_lags=(vd_reference_lag,), neutral)
        ref = init(reference, MethodOfSteps(RT.Tsit5()); dt=1/200, dtmax=0.01,
            dtmin=0.0, adaptive=true, abstol=1e-12, reltol=1e-10,
            save_on=false, save_start=false, save_end=false, save_everystep=false,
            dense=false, maxiters=typemax(Int))
        out = zeros(Float32, 20)
        retained = source.data.integrator
        slots = objectid.(retained.integrator.sol.u.slots)
        for block in 0:159
            for (time, _, target, _) in VD_SCHEDULE
                block == round(Int, 10time) && set_param!(source, 1, target)
            end
            render!(out, source)
            vd_advance_reference!(ref, (block+1)/10)
            @test source.data.integrator === retained
            @test !source.data.renderer.failed
            @test retained.u ≈ ref.u atol=2e-6 rtol=2e-6
            @test length(retained.integrator.sol.t) <= 256
            @test length(retained.tracked_discontinuities) <= source.data.delay_window.discontinuity_capacity
            for sample in (1, 10, 20)
                t = block/10+(sample-1)/200
                @test out[sample] ≈ Float32(ref.history(ref.p, t; idxs=1)) atol=3e-6 rtol=3e-6
                !neutral && t < 0.2 && @test out[sample] ≈ Float32(0.52+0.375t) atol=2e-7
                neutral && @test out[sample] ≈ Float32(sin(8t)) atol=3e-6
            end
            if block > 10
                t = retained.t-0.65
                @test retained.history(retained.p, t; idxs=1) ≈ ref.history(ref.p, t; idxs=1) atol=3e-6
                neutral && @test retained.history(retained.p, t, Val{1}; idxs=1) ≈
                    ref.history(ref.p, t, Val{1}; idxs=1) atol=3e-5 rtol=3e-6
            end
        end
        @test retained.stats.naccept > 4*256
        @test objectid.(retained.integrator.sol.u.slots) == slots
        @test length(ref.integrator.sol.t) > 4length(retained.integrator.sol.t)
    end
end

vd_parameter_history(p, t; idxs=nothing) = idxs === nothing ? [p[1]] : p[1]
vd_parameter_history(out, p, t) = (fill!(out, p[1]); nothing)
vd_history_rhs!(du, u, h, p, t) = (du[1] = h(p, t-p[1]; idxs=1); nothing)
vd_history_inplace_rhs!(du, u, h, p, t) = (h(du, p, t-p[1]); nothing)

@testset "Initial history stays fixed while tau changes" begin
    for f in (vd_history_rhs!, vd_history_inplace_rhs!)
        prob = DDEProblem(f, [0.5], vd_parameter_history, (0.0, 5.0), [0.5])
        source = DESource(prob; delay_parameter=1, min_delay=0.5, max_delay=1.0,
            dtmax=0.02, history_capacity=128, channel_map=[1])
        prepare!(source; sample_rate=400, buffer_size=20)
        out = zeros(Float32, 20)
        render!(out, source)
        set_param!(source, 1, 1.0)
        for block in 1:4
            render!(out, source)
            @test out ≈ Float32[0.5+0.5*(block/20+(j-1)/400) for j in 1:20] atol=2e-7
        end
        reset_state!(source)
        render!(out, source)
        @test out ≈ Float32[0.5+(j-1)/400 for j in 1:20] atol=2e-7
    end
end

vd_observe_delay(u, p, t, h) = (p[1],)

@testset "Delay controls remain continuous across retarget, prepare and reset" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 10.0), [0.2, 1.0])
    source = DESource(prob; delay_parameter=1, min_delay=0.1, max_delay=0.7,
        delay_smoothing=0.05, dtmax=0.01, history_capacity=256, channel_map=[1])
    prepare!(source; sample_rate=200, buffer_size=20)
    retained = source.data.integrator
    out = zeros(Float32, 20)
    set_param!(source, 1, 0.7)
    prepare!(source; sample_rate=200, buffer_size=20)
    @test source.data.integrator === retained
    render!(out, source; observe=vd_observe_delay)
    @test out[1] ≈ 0.2f0
    @test 0.2 < out[end] < 0.7
    @test all(>=(0), diff(out))
    current = RT._delay_value(source.data.delay_window.delay, retained.t)
    set_param!(source, 1, 0.1)
    render!(out, source; observe=vd_observe_delay)
    @test out[1] ≈ Float32(current)
    @test all(<=(0), diff(out))
    @test all(x -> 0.1 <= x <= 0.7, out)
    reset_state!(source)
    render!(Float32[], source)
    @test retained.t == 0.0
    @test length(retained.integrator.sol.t) == 1
    render!(out, source; observe=vd_observe_delay)
    @test all(==(0.1f0), out)
    set_param!(source, 2, 2.0)
    @test_throws ArgumentError render!(out, source)
    reset_state!(source)
    render!(out, source)
    @test !source.data.renderer.failed
end

@testset "Delay smoothing uses audio seconds" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 5.0), [0.2])
    source = DESource(prob; delay_parameter=1, min_delay=0.1, max_delay=0.3,
        delay_smoothing=0.05, dtmax=0.01, history_capacity=128, channel_map=[1])
    set_ts!(source, 20.0)
    prepare!(source; sample_rate=1000, buffer_size=25)
    set_param!(source, 1, 0.21)
    out = zeros(Float32, 25)
    render!(out, source; observe=vd_observe_delay)
    @test out[1] ≈ 0.2f0
    render!(out, source; observe=vd_observe_delay)
    @test out[1] ≈ 0.205f0
    render!(out, source; observe=vd_observe_delay)
    @test all(==(0.21f0), out)
    @test !source.data.renderer.failed
end

function vd_render_bytes!(source, out, repetitions)
    @allocated for _ in 1:repetitions
        render!(out, source)
    end
end

function vd_cycle_bytes!(source, out, cycles)
    bytes = 0
    for cycle in 1:cycles
        set_param!(source, 1, isodd(cycle) ? 0.6 : 0.5)
        bytes += vd_render_bytes!(source, out, 25)
    end
    bytes
end

@testset "Moving-delay rendering reuses storage" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 100.0), [0.5])
    source = DESource(prob; delay_parameter=1, min_delay=0.5, max_delay=4.0,
        dtmax=0.01, history_capacity=2048, channel_map=[1])
    prepare!(source; sample_rate=1000, buffer_size=16)
    out = zeros(Float32, 16)
    empty = Float32[]
    set_param!(source, 1, 0.6)
    vd_render_bytes!(source, empty, 1)
    set_param!(source, 1, 0.5)
    reset_state!(source)
    vd_render_bytes!(source, empty, 1)
    vd_render_bytes!(source, out, 300)
    set_param!(source, 1, 4.0)
    @test vd_render_bytes!(source, empty, 1) == 0
    vd_render_bytes!(source, out, 50)
    @test vd_render_bytes!(source, out, 50) == 0
    @test !source.data.renderer.failed
    delay = source.data.delay_window.delay
    @test delay.origin < source.data.integrator.t < delay.origin+delay.duration
end

@testset "Completed delay ramps preserve reusable event storage" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 100.0), [0.5])
    source = DESource(prob; delay_parameter=1, min_delay=0.5, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    prepare!(source; sample_rate=1000, buffer_size=16)
    out = zeros(Float32, 16)
    vd_cycle_bytes!(source, out, 20)
    @test vd_cycle_bytes!(source, out, 20) == 0
    @test !source.data.renderer.failed
    delay = source.data.delay_window.delay
    @test source.data.integrator.t >= delay.origin+delay.duration
    @test RT._delay_value(delay, source.data.integrator.t) ≈ 0.5
end

@testset "Variable-delay capacity exhaustion silences output" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 5.0), [0.5])
    for (history_capacity, discontinuity_capacity) in ((4, 32), (512, 4))
        source = DESource(prob; delay_parameter=1, min_delay=0.1, max_delay=1.0,
            dtmax=0.01, history_capacity, discontinuity_capacity, channel_map=[1])
        prepare!(source; sample_rate=1000, buffer_size=10)
        out = ones(Float32, 10)
        for block in 1:20
            set_param!(source, 1, isodd(block) ? 0.6 : 0.7)
            render!(out, source)
            source.data.renderer.failed && break
        end
        @test source.data.renderer.failed
        @test all(iszero, out)
        @test length(source.data.integrator.integrator.sol.t) <= history_capacity
        @test length(source.data.integrator.tracked_discontinuities) <= discontinuity_capacity
        reset_state!(source)
        render!(Float32[], source)
        @test !source.data.renderer.failed
    end
end
