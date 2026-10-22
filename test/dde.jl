using DelayDiffEq: MethodOfSteps
using OrdinaryDiffEqRosenbrock: Rodas5P
using SciMLBase: DDEProblem, init, step!, set_proposed_dt!

initial_history(p, t; idxs=nothing) = idxs === nothing ? [sin(t)] : sin(t)
initial_history(p, t, ::Type{Val{1}}; idxs=nothing) = idxs === nothing ? [cos(t)] : cos(t)

function delayed_sine!(du, u, h, p, t)
    du[1] = cos(t) - p[1]*(u[1] - sin(t)) +
        0.3*(h(p, t-p[2]; idxs=1) - sin(t-p[2]))
end

function neutral_sine!(du, u, h, p, t)
    du[1] = cos(t) - p[1]*(u[1] - sin(t)) +
        0.3*(h(p, t-p[2], Val{1}; idxs=1) - cos(t-p[2]))
end

@testset "SciML DDE interpolation survives rolling history" begin
    for neutral in (false, true), alg in (MethodOfSteps(RT.Tsit5()), MethodOfSteps(Rodas5P()))
        f = neutral ? neutral_sine! : delayed_sine!
        t0, dt = 0.3, 1/800
        p = [alg.alg isa Rodas5P ? 1e4 : 1.0, 0.137]
        prob = DDEProblem(f, [sin(t0)], initial_history, (t0, 10.0), p;
            constant_lags=(p[2],), neutral)
        source = DESource(prob; alg, channel_map=[1])
        prepare!(source; sample_rate=400, buffer_size=64, substeps=2)
        ref = init(remake(prob; tspan=(t0, Inf)), alg; dt, dtmax=dt, dtmin=0.0, adaptive=true,
            abstol=1e-12, reltol=1e-10,
            save_on=false, save_start=false, save_end=false, save_everystep=false,
            dense=false, maxiters=typemax(Int))
        buffer = zeros(Float32, 64)
        steps = 0
        for block in 1:30
            render!(buffer, source)
            for _ in 1:128
                steps += 1
                target = t0 + steps*dt
                step!(ref, target-ref.t, true)
            end
            integ = source.data.integrator
            @test !source.data.renderer.failed
            @test integ.t ≈ ref.t atol=1e-13
            @test integ.u ≈ ref.u atol=1e-10 rtol=1e-10
            @test integ.u[1] ≈ sin(integ.t) atol=1e-5
            saved = integ.integrator.sol
            @test length(saved.t) < 500
            @test length(saved.t) == length(saved.u) == length(saved.k)
            @test isempty(integ.sol.t)
            @test isempty(integ.tracked_discontinuities)
            for delay in (0.002, 0.073, p[2]), derivative in (Val{0}, Val{1})
                t = integ.t-delay
                @test integ.history(p, t, derivative; idxs=1) ≈
                    ref.history(p, t, derivative; idxs=1) atol=2e-8 rtol=2e-8
            end
        end
        retained = source.data.integrator
        prepare!(source; sample_rate=400, buffer_size=64, substeps=2)
        @test source.data.integrator === retained
        render!(buffer, source)
        @test !source.data.renderer.failed
        reset_state!(source)
        render!(Float32[], source)
        @test source.data.integrator.t == t0
        @test length(source.data.integrator.integrator.sol.t) == 1
        render!(buffer, source; observe=(u,p,t,h)->(h(p,t-p[2]; idxs=1),))
        @test buffer[1] ≈ Float32(sin(t0-p[2])) atol=1e-6
        @test !source.data.renderer.failed
        set_param!(source, 1, p[1]*2)
        @test_throws ArgumentError render!(buffer, source)
        reset_state!(source)
        render!(buffer, source)
        @test !source.data.renderer.failed
    end
end

@testset "DDE validation" begin
    prob = DDEProblem(delayed_sine!, [0.0], initial_history, (0.0,1.0), [1.0,0.1];
        constant_lags=(0.1,))
    @test_throws ArgumentError DESource(prob; alg=RT.Tsit5())
    @test_throws ArgumentError DESource(prob; max_delay=0.01)
    @test_throws ArgumentError DESource(remake(prob; constant_lags=()))
    @test_throws ArgumentError DESource(remake(prob; constant_lags=(-0.1,)))
    @test_throws ArgumentError DESource(remake(prob; neutral=true, constant_lags=(0.1,0.2)))
    @test_throws ArgumentError prepare!(DESource(prob); sample_rate=1)
end

@testset "Neutral echoes and sustained bounded storage" begin
    history(p,t; idxs=nothing) = idxs === nothing ? [0.0] : 0.0
    history(p,t,::Type{Val{1}}; idxs=nothing) = history(p,t; idxs)
    echo!(du,u,h,p,t) = (du[1] = 1 + p[1]*h(p,t-p[2],Val{1}; idxs=1); nothing)
    p = [0.4,0.1]
    prob = DDEProblem(echo!, [0.0], history, (0.0,100.0), p;
        constant_lags=(p[2],), neutral=true)
    source = DESource(prob; channel_map=[1])
    prepare!(source; sample_rate=400, buffer_size=40)
    buffer = zeros(Float32,40)
    retained = Int[]
    for block in 1:200
        render!(buffer,source)
        @test !source.data.renderer.failed
        for frame in (11,31)
            n = block-1
            remainder = (frame-1)/400
            slope = (1-p[1]^(n+1))/(1-p[1])
            integral = p[2]*(n/(1-p[1])-p[1]*(1-p[1]^n)/(1-p[1])^2)
            @test buffer[frame] ≈ Float32(integral + remainder*slope) atol=3e-6
        end
        integ=source.data.integrator
        @test length(integ.integrator.sol.t) <= 500
        @test length(integ.tstops_propagated) <= 1
        @test length(integ.d_discontinuities_propagated) <= 1
        block in (20,100,200) && push!(retained,Base.summarysize(integ))
    end
    @test maximum(retained) < 500_000
end

@testset "DDE history limit silences output" begin
    prob = DDEProblem(delayed_sine!, [0.0], initial_history, (0.0,1.0), [1.0,0.1];
        constant_lags=(0.1,))
    source = DESource(prob; channel_map=[1], history_capacity=4)
    prepare!(source; sample_rate=400, buffer_size=40)
    buffer = ones(Float32,40)
    render!(buffer,source)
    @test source.data.renderer.failed
    @test all(iszero,buffer)
    @test length(source.data.integrator.integrator.sol.t) <= 4
    render!(buffer,source)
    @test all(iszero,buffer)
    reset_state!(source)
    render!(Float32[],source)
    @test !source.data.renderer.failed
    @test length(source.data.integrator.integrator.sol.t) == 1
end
