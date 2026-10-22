using Distributions, RealTimeAudioDiffEq

function nreed!(du, u, p, t)
    ntot = length(u) ÷ 2
    for n in 1:ntot
        μ = p[3n - 1]
        k = p[3n]
        v0 = p[3n + 1]
        v = u[2n] - v0
        previous = n > 1 ? 2n - 3 : 2ntot - 1
        du[2n - 1] = u[2n] + p[1] * u[previous]
        du[2n] = v * (μ - v * v) - k * u[2n - 1]
    end
    nothing
end

function nreeds(ntot,kmean,vmean,dmu)
    k = rand(Gamma(5, kmean/4.0),ntot)
	v0 = rand(Gamma(5, vmean/4.0),ntot)
	u0 = repeat([0.1,0.],ntot)
	p = vcat([0.0],[[3*v0[n]^2+dmu,k[n],v0[n]] for n=1:ntot]...)
    mix = zeros(2*ntot,2)
	mix[1:2:2*ntot,1] = range(0, 1, ntot)
	mix[1:2:2*ntot,2] = range(1, 0, ntot)
    source = DESource(nreed!, u0, p; channel_map = mix)
    println(source)
    println("Starting audio...")
    output_device = get_default_output_device()
    set_ts!(source,1500.0)
	set_gain!(source,0.2)
	start_DESource(source, output_device; buffer_size=convert(UInt32,1024))
    return source
end;

function test_nreed(nosc=8,kmean=1.3,vmean=0.3,dmu=0.1,dur=1.0)
    source = nreeds(nosc,kmean,vmean,dmu)
    sleep(dur)
    stop_DESource(source)
end



