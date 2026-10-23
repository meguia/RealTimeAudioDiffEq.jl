@testset "Native adaptive sampling on the audio clock" begin
    for neutral in (false, true), alg in (MethodOfSteps(RT.Tsit5()), MethodOfSteps(Rodas5P()))
        f = neutral ? neutral_sine! : delayed_sine!
        t0, lag, fs = 0.3, 0.137, 1000
        p = [alg.alg isa Rodas5P ? 1e4 : 1.0, lag]
        prob = DDEProblem(f, [sin(t0)], initial_history, (t0, 10.0), p;
            constant_lags=(lag,), neutral)
        source = DESource(prob; alg, channel_map=[1], dtmax=lag/4, history_capacity=512)
        prepare!(source; sample_rate=fs, buffer_size=64, substeps=2)
        buffer = zeros(Float32, 64)
        produced = 0
        for block in 1:30
            render!(buffer, source)
            times = t0 .+ ((produced:produced+63) ./ fs)
            @test buffer ≈ Float32.(sin.(times)) atol=2e-6
            produced += 64
            integ = source.data.integrator
            saved = integ.integrator.sol
            @test !source.data.renderer.failed
            @test integ.t ≈ t0+produced/fs atol=1e-13
            @test length(saved.t) == length(saved.u) == length(saved.k)
            @test first(saved.t) <= integ.t-lag || first(saved.t) == t0
            @test isempty(integ.tracked_discontinuities)
        end
        alg.alg isa RT.Tsit5 && @test source.data.integrator.stats.naccept < produced
        observe(u,p,t,h) = (h(p,t-p[2],Val{1}; idxs=1),)
        before = source.data.integrator.t
        render!(buffer, source; observe)
        @test buffer ≈ Float32.(cos.(before-lag .+ (0:63)./fs)) atol=2e-6
        retained = source.data.integrator
        prepare!(source; sample_rate=fs, buffer_size=64, substeps=2)
        @test source.data.integrator === retained
        reset_state!(source)
        render!(Float32[], source)
        @test source.data.integrator.t == t0
        @test length(source.data.integrator.integrator.sol.t) == 1
        render!(@view(buffer[1:7]), source)
        @test buffer[1:7] ≈ Float32.(sin.(t0 .+ (0:6)./fs)) atol=2e-6
        @test source.data.integrator.t ≈ t0+7/fs atol=1e-13
    end
end

@testset "Adaptive sampling validation and bounded-history failure" begin
    prob = DDEProblem(delayed_sine!, [0.0], initial_history, (0.0,1.0), [1.0,0.1];
        constant_lags=(0.1,))
    for dtmax in (0.0,-0.1,NaN,Inf,0.2)
        @test_throws ArgumentError DESource(prob; dtmax)
    end
    source = DESource(prob; channel_map=[1], dtmax=0.01, history_capacity=4)
    prepare!(source; sample_rate=1000, buffer_size=64)
    buffer = ones(Float32,64)
    render!(buffer,source)
    @test source.data.renderer.failed
    @test all(iszero,buffer)
    @test length(source.data.integrator.integrator.sol.t) <= 4
end

@testset "Native sampling preserves neutral echo jumps" begin
    history(p,t; idxs=nothing) = idxs === nothing ? [0.0] : 0.0
    history(p,t,::Type{Val{1}}; idxs=nothing) = history(p,t; idxs)
    echo!(du,u,h,p,t) = (du[1] = 1+p[1]*h(p,t-p[2],Val{1}; idxs=1); nothing)
    p = [0.4,0.1]
    prob = DDEProblem(echo!, [0.0], history, (0.0,10.0), p;
        constant_lags=(p[2],), neutral=true)
    source = DESource(prob; channel_map=[1,1], dtmax=0.025, history_capacity=512)
    prepare!(source; sample_rate=400, buffer_size=40)
    buffer = zeros(Float32,80)
    observe(u,p,t,h) = (u[1],h(p,t-p[2],Val{1}; idxs=1))
    for block in 1:50
        render!(buffer,source; observe)
        @test !source.data.renderer.failed
        n = block-1
        slope = (1-p[1]^(n+1))/(1-p[1])
        integral = p[2]*(n/(1-p[1])-p[1]*(1-p[1]^n)/(1-p[1])^2)
        delayed = (1-p[1]^n)/(1-p[1])
        for frame in (11,31)
            @test buffer[2frame-1] ≈ Float32(integral+(frame-1)/400*slope) atol=3e-6
            @test buffer[2frame] ≈ Float32(delayed) atol=3e-6
        end
        @test length(source.data.integrator.tstops_propagated) <= 1
    end
end
