reuse_history(p, t; idxs=nothing) = idxs === nothing ? [sin(p[3]*t)] : sin(p[3]*t)
reuse_history(p, t, ::Type{Val{1}}; idxs=nothing) =
    idxs === nothing ? [p[3]*cos(p[3]*t)] : p[3]*cos(p[3]*t)

function reuse_retarded!(du, u, h, p, t)
    du[1] = p[3]*cos(p[3]*t) - p[1]*(u[1]-sin(p[3]*t)) +
        0.2*(h(p,t-p[2]; idxs=1)-sin(p[3]*(t-p[2])))
    nothing
end

@testset "History reuse is isolated and optional" begin
    prob = DDEProblem(delayed_sine!,[0.0],initial_history,(0.0,10.0),[1.0,0.137];
        constant_lags=(0.137,))
    unmarked = init(prob,MethodOfSteps(RT.Tsit5()); dt=0.001)
    @test unmarked.integrator.sol.u isa Vector
    @test unmarked.integrator.sol.k isa Vector
    reused = DESource(prob; channel_map=[1],dtmax=0.01,history_capacity=128)
    original = DESource(prob; channel_map=[1],dtmax=0.01,history_capacity=128,reuse_history=false)
    for source in (reused,original)
        prepare!(source; sample_rate=1000,buffer_size=32)
    end
    @test reused.data.integrator.integrator.sol.u isa RT.HistoryBuffer
    @test reused.data.integrator.integrator.sol.k isa RT.HistoryBuffer
    @test original.data.integrator.integrator.sol.u isa Vector
    @test original.data.integrator.integrator.sol.k isa Vector
    before_u = objectid.(reused.data.integrator.integrator.sol.u.slots)
    a,b = zeros(Float32,32),zeros(Float32,32)
    for _ in 1:100
        render!(a,reused)
        render!(b,original)
        @test a == b
        @test !reused.data.renderer.failed && !original.data.renderer.failed
    end
    @test objectid.(reused.data.integrator.integrator.sol.u.slots) == before_u
    @test reused.data.integrator.stats.naccept == original.data.integrator.stats.naccept
end

function reuse_neutral!(du, u, h, p, t)
    du[1] = p[3]*cos(p[3]*t) - p[1]*(u[1]-sin(p[3]*t)) +
        0.2*(h(p,t-p[2],Val{1}; idxs=1)-p[3]*cos(p[3]*(t-p[2])))
    nothing
end

function reuse_render_bytes!(source, buffer, n)
    @allocated for _ in 1:n
        render!(buffer, source)
    end
end

@testset "Repeated history reuse preserves native values and derivatives" begin
    for neutral in (false, true), alg in (MethodOfSteps(RT.Tsit5()), MethodOfSteps(Rodas5P()))
        t0, lag, omega, fs, frames, capacity = 0.3, 0.031, 80.0, 200, 16, 512
        p = [alg.alg isa Rodas5P ? 1e4 : 1.0, lag, omega]
        prob = DDEProblem(neutral ? reuse_neutral! : reuse_retarded!, [sin(omega*t0)],
            reuse_history, (t0,100.0), p; constant_lags=(lag,), neutral)
        source = DESource(prob; alg, channel_map=[1,1], dtmax=lag/4,
            history_capacity=capacity)
        prepare!(source; sample_rate=fs, buffer_size=frames)
        ref = init(remake(prob; tspan=(t0,Inf)), alg; dt=1/fs,dtmax=lag/4,dtmin=0.0,
            adaptive=true,abstol=1e-12,reltol=1e-10,save_on=false,save_start=false,
            save_end=false,save_everystep=false,dense=false,maxiters=typemax(Int))
        buffer = zeros(Float32,2frames)
        observe(u,p,t,h) = (u[1],h(p,t-p[2],Val{1}; idxs=1))
        for block in 1:100
            render!(buffer,source; observe)
            endpoint = t0+block*frames/fs
            RT.SciMLBase.add_tstop!(ref,endpoint)
            while ref.t < endpoint
                step!(ref)
            end
            integ = source.data.integrator
            @test !source.data.renderer.failed
            @test integ.t ≈ endpoint atol=1e-12
            @test integ.u ≈ ref.u atol=1e-8 rtol=1e-8
            @test length(integ.integrator.sol.t) <= capacity
            for frame in (2,8,15)
                t = t0+((block-1)*frames+frame-1)/fs
                @test buffer[2frame-1] ≈ Float32(sin(omega*t)) atol=2e-6
                @test buffer[2frame] ≈ Float32(omega*cos(omega*(t-lag))) atol=1e-4
            end
            for fraction in (0.1,0.55,1.0), derivative in (Val{0},Val{1})
                t = integ.t-fraction*lag
                @test integ.history(p,t,derivative; idxs=1) ≈
                    ref.history(p,t,derivative; idxs=1) atol=2e-6 rtol=1e-8
            end
        end
        @test source.data.integrator.stats.naccept > 4capacity
        @test source.data.integrator.stats.nreject > 0
        retained = source.data.integrator
        prepare!(source; sample_rate=fs,buffer_size=frames)
        @test source.data.integrator === retained
        reset_state!(source)
        render!(Float32[],source)
        @test source.data.integrator.t == t0
        @test length(source.data.integrator.integrator.sol.t) == 1
        render!(buffer,source; observe)
        @test !source.data.renderer.failed
        @test buffer[1] ≈ Float32(sin(omega*t0)) atol=2e-6
    end
end

@testset "Warm Tsit5 history reuse allocations" begin
    for neutral in (false,true)
        prob = DDEProblem(neutral ? neutral_sine! : delayed_sine!, [0.0],initial_history,
            (0.0,100.0),[1.0,0.137]; constant_lags=(0.137,),neutral)
        source = DESource(prob; channel_map=[1],dtmax=0.01,history_capacity=512)
        prepare!(source; sample_rate=1000,buffer_size=64)
        buffer = zeros(Float32,64)
        reuse_render_bytes!(source,buffer,200)
        @test reuse_render_bytes!(source,buffer,100) == 0
        @test !source.data.renderer.failed
        @test all(isfinite,buffer)
    end
end
