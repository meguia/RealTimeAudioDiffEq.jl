using RealTimeAudioDiffEq
using Test

@testset "RealTimeAudioDiffEq.jl" begin
    function ode_rhs!(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = -p[2] * u[2]
        return nothing
    end

    function sde_drift!(du, u, p, t)
        du[1] = -p[1] * u[1]
        return nothing
    end

    function sde_noise!(du, u, p, t)
        du[1] = p[2]
        return nothing
    end

    @testset "ODE constructor and controls" begin
        u0 = [0.1, 0.2]
        p = [2.0, 3.0]
        mix = [1.0 0.0; 0.0 1.0]

        src = DESource(ode_rhs!, u0, p; channel_map = mix)

        @test get_u0(src) == u0
        @test get_params(src) == p
        @test get_param(src, 1) == 2.0
        @test get_ts(src) == 1.0
        @test get_gain(src) == 1.0
        @test get_channelmap(src) == mix

        set_ts!(src, 1200.0)
        @test get_ts(src) == 1200.0

        set_gain!(src, 0.25)
        @test get_gain(src) == 0.25

        set_param!(src, 2, 4.5)
        @test get_param(src, 2) == 4.5

        set_u0!(src, [0.9, -0.3])
        @test get_u0(src) == [0.9, -0.3]

        new_mix = [0.5 0.5; 0.5 0.5]
        set_channelmap!(src, new_mix)
        @test get_channelmap(src) == new_mix

        @test_throws ErrorException set_param!(src, 99, 1.0)
        @test_throws ErrorException get_param(src, 99)
        @test_throws ErrorException set_u0!(src, [1.0])
    end

    @testset "channel_map backward compatibility" begin
        u0 = [0.1, 0.2]
        p = [2.0, 3.0]

        src_idx = DESource(ode_rhs!, u0, p; channel_map = [1, 2])
        @test get_channelmap(src_idx) == [1.0 0.0; 0.0 1.0]

        src_groups = DESource(ode_rhs!, u0, p; channel_map = [[1], [1, 2]])
        @test get_channelmap(src_groups) == [1.0 1.0; 0.0 1.0]

        set_channelmap!(src_idx, [2, 1])
        @test get_channelmap(src_idx) == [0.0 1.0; 1.0 0.0]

        @test_throws ErrorException DESource(ode_rhs!, u0, p; channel_map = [3])
        @test_throws ErrorException set_channelmap!(src_idx, [[0]])
    end

    @testset "SDE constructor" begin
        u0 = [0.5]
        p = [0.2, 0.1]
        mix = [1.0;;]

        src = DESource(sde_drift!, sde_noise!, u0, p; channel_map = mix)

        @test get_u0(src) == u0
        @test get_params(src) == p
        @test get_channelmap(src) == mix
    end
end
