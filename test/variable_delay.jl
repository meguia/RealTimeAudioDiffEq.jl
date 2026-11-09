using DelayDiffEq: MethodOfSteps
using SciMLBase: DDEProblem, init, step!, derivative_discontinuity!

vd_lag(u, p, t) = p[1]
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

function vd_scheduled_lag(u, p, t)
    value = 0.2
    for (origin, before, after, duration) in VD_SCHEDULE
        t < origin && return value
        if t < origin+duration
            x = (t-origin)/duration
            return clamp(before + (after-before)*(3x^2-2x^3), min(before, after), max(before, after))
        end
        value = after
    end
    value
end

struct VDWithLag{F,L}
    equation::F
    lag::L
    parameters::Vector{Float64}
end

function (f::VDWithLag)(du, u, h, p, t)
    f.parameters[1] = f.lag(u, p, t)
    f.equation(du, u, h, f.parameters, t)
end

function vd_advance_reference!(integ, target)
    RT.SciMLBase.add_tstop!(integ, target)
    while integ.t < target
        previous = integ.t
        step!(integ)
        integ.t > previous || error("reference solver did not advance")
    end
end

@testset "External delay validation and ownership" begin
    p = [0.5]
    prob = DDEProblem(vd_cubic!, [0.52], vd_constant_history, (0.0, 20.0), p;
        dependent_lags=(vd_lag,))
    options = (min_delay=0.1, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    source = DESource(prob; options...)
    @test isempty(source.data.problem.constant_lags)
    @test length(source.data.problem.dependent_lags) == 1
    p[1] = 0.6
    @test get_param(source, 1) == 0.5
    for bad in ((min_delay=0.0,), (min_delay=-0.1,), (min_delay=NaN,),
            (max_delay=Inf,), (max_delay=0.4,), (min_delay=0.6,),
            (min_delay=nothing,), (max_delay=nothing,), (dtmax=0.11,),
            (discontinuity_capacity=3,))
        @test_throws ArgumentError DESource(remake(prob; p=[0.5]); merge(options, bad)...)
    end
    @test_throws ArgumentError DESource(remake(prob; constant_lags=(0.5,)); options...)
    @test_throws ArgumentError DESource(remake(prob; dependent_lags=(vd_lag, vd_lag)); options...)
    @test_throws ArgumentError DESource(prob; options..., alg=MethodOfSteps(Rodas5P()))
    for value in (0.0, 0.099, 0.701, Inf, NaN)
        @test_throws ArgumentError DESource(remake(prob; p=[value]); options...)
    end
    for value in (Inf, NaN)
        @test_throws ArgumentError set_param!(source, 1, value)
        @test get_param(source, 1) == 0.5
    end
    @test_throws ArgumentError prepare!(source; sample_rate=5)
    for value in (0.1, 0.7)
        set_param!(source, 1, value)
        prepare!(source; sample_rate=200)
        render!(zeros(Float32, 20), source)
        @test source.data.integrator.p[1] == value
        @test !source.data.renderer.failed
    end
    set_param!(source, 1, 0.8)
    @test_throws ArgumentError render!(zeros(Float32, 20), source)

    late_lag(u, p, t) = t < 0.05 ? 0.5 : 0.8
    late_prob = DDEProblem(VDWithLag(vd_cubic!, late_lag, [0.5]), [0.52],
        vd_constant_history, (0.0, 1.0), [0.5]; dependent_lags=(late_lag,))
    late = DESource(late_prob; options...)
    prepare!(late; sample_rate=1000)
    @test_throws ArgumentError render!(zeros(Float32, 100), late)
end

@testset "External cubic and neutral delays match unbounded native integration" begin
    for neutral in (false, true)
        f = neutral ? vd_neutral! : vd_cubic!
        history = neutral ? vd_sine_history : vd_constant_history
        initial = neutral ? [0.0] : [0.52]
        prob = DDEProblem(VDWithLag(f, vd_scheduled_lag, [0.2]), initial, history,
            (0.0, Inf), [0.2]; dependent_lags=(vd_scheduled_lag,), neutral)
        source = DESource(prob; min_delay=0.1, max_delay=0.7,
            dtmax=0.01, history_capacity=256, channel_map=[1])
        prepare!(source; sample_rate=200, buffer_size=20)
        reference = remake(prob; f=VDWithLag(f, vd_scheduled_lag, [0.2]))
        ref = init(reference, MethodOfSteps(RT.Tsit5()); dt=1/200, dtmax=0.01,
            dtmin=0.0, adaptive=true, abstol=1e-12, reltol=1e-10,
            save_on=false, save_start=false, save_end=false, save_everystep=false,
            dense=false, maxiters=typemax(Int))
        out = zeros(Float32, 20)
        retained = source.data.integrator
        slots = objectid.(retained.integrator.sol.u.slots)
        for block in 0:159
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

@testset "Initial history stays fixed until reset" begin
    for f in (vd_history_rhs!, vd_history_inplace_rhs!), reuse_history in (false, true)
        prob = DDEProblem(f, [0.5], vd_parameter_history, (0.0, 5.0), [0.5];
            dependent_lags=(vd_lag,))
        source = DESource(prob; min_delay=0.5, max_delay=1.0, reuse_history,
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

@testset "Parameter updates have no implicit smoothing" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 10.0), [0.2, 1.0];
        dependent_lags=(vd_lag,))
    source = DESource(prob; min_delay=0.1, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    prepare!(source; sample_rate=200, buffer_size=20)
    retained = source.data.integrator
    out = zeros(Float32, 20)
    set_param!(source, 1, 0.7)
    prepare!(source; sample_rate=200, buffer_size=20)
    @test source.data.integrator === retained
    render!(out, source; observe=vd_observe_delay)
    @test all(==(0.7f0), out)
    set_param!(source, 1, 0.1)
    render!(out, source; observe=vd_observe_delay)
    @test all(==(0.1f0), out)
    set_param!(source, 2, 2.0)
    render!(out, source)
    @test !source.data.renderer.failed
    @test retained.p[2] == 2.0
    reset_state!(source)
    render!(Float32[], source)
    @test retained.t == 0.0
    @test length(retained.integrator.sol.t) == 1
    render!(out, source; observe=vd_observe_delay)
    @test all(==(0.1f0), out)
end

mutable struct VDExternalRamp
    position::Float64
    origin::Float64
    rate::Int
end
function (r::VDExternalRamp)(u, p, t)
    x = clamp((r.position+r.rate*(t-r.origin))/0.1, 0.0, 1.0)
    0.2 + 0.4*x*x*(3-2x)
end
function vd_rate!(r, integ, rate)
    r.position = clamp(r.position+r.rate*(integ.t-r.origin), 0.0, 0.1)
    r.origin, r.rate = integ.t, rate
    derivative_discontinuity!(integ, true)
end

@testset "Application smoothing can pause and reverse without a second ramp" begin
    ramp = VDExternalRamp(0.0, 0.0, 1)
    prob = DDEProblem(VDWithLag(vd_neutral!, ramp, [0.2]), [0.0],
        vd_sine_history, (0.0, Inf), [0.2]; dependent_lags=(ramp,), neutral=true)
    source = DESource(prob; min_delay=0.1, max_delay=0.7,
        dtmax=0.002, history_capacity=512, channel_map=[1])
    prepare!(source; sample_rate=1000, buffer_size=10)
    integ = source.data.integrator
    out = zeros(Float32, 10)
    observe(u, p, t, h) = (ramp(u, p, t),)
    for (rate, blocks) in ((1, 4), (0, 2), (-1, 3), (1, 10))
        vd_rate!(ramp, integ, rate)
        render!(Float32[], source)
        @test any(e -> e.t == integ.t && e.order <= 1, integ.tracked_discontinuities)
        events = length(integ.tracked_discontinuities)
        derivative_discontinuity!(integ, true)
        render!(Float32[], source)
        @test length(integ.tracked_discontinuities) == events
        for _ in 1:blocks
            t0 = integ.t
            expected = Float32[ramp(integ.u, integ.p, t0+(j-1)/1000) for j in 1:10]
            render!(out, source; observe)
            @test out ≈ expected atol=2e-7
            @test integ.u[1] ≈ sin(8integ.t) atol=3e-6
            @test !source.data.renderer.failed
        end
    end
    @test out[end] == 0.6f0
    @test integ.p == [0.2]
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

@testset "External moving delay reuses history storage" begin
    lag(u, p, t) = 0.55 + 0.05sin(t)
    prob = DDEProblem(VDWithLag(vd_sine!, lag, [0.5]), [0.0], vd_sine_history,
        (0.0, Inf), [0.5]; dependent_lags=(lag,))
    source = DESource(prob; min_delay=0.5, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    prepare!(source; sample_rate=1000, buffer_size=16)
    out = zeros(Float32, 16)
    vd_render_bytes!(source, out, 300)
    @test vd_render_bytes!(source, out, 50) == 0
    @test !source.data.renderer.failed
end

@testset "Repeated controls preserve reusable event storage" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, Inf), [0.5];
        dependent_lags=(vd_lag,))
    source = DESource(prob; min_delay=0.5, max_delay=0.7,
        dtmax=0.01, history_capacity=256, channel_map=[1])
    prepare!(source; sample_rate=1000, buffer_size=16)
    out = zeros(Float32, 16)
    vd_cycle_bytes!(source, out, 20)
    @test vd_cycle_bytes!(source, out, 20) == 0
    @test !source.data.renderer.failed
    @test all(e -> e.t >= source.data.integrator.t-0.7,
        source.data.integrator.tracked_discontinuities)
end

@testset "Variable-delay capacity exhaustion silences output" begin
    prob = DDEProblem(vd_sine!, [0.0], vd_sine_history, (0.0, 5.0), [0.5];
        dependent_lags=(vd_lag,))
    for (history_capacity, discontinuity_capacity) in ((4, 32), (512, 4))
        source = DESource(prob; min_delay=0.1, max_delay=1.0,
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
