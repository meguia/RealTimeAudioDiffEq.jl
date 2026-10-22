# RealTimeAudioDiffEq

Real-time audio from ordinary and stochastic differential equations (ODEs and
SDEs), using SciML's persistent integrator interface.

This branch replaces per-block solving with an initialized integrator advanced by `step!`.

## Install and run

For a local checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/RealTimeAudioDiffEq.jl")
```

Start Julia with at least two threads for playback (`julia --threads=2`). Offline
rendering also works with one thread.

```julia
using RealTimeAudioDiffEq

function oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1] - p[2] * u[2]
end

source = DESource(oscillator!, [0.1, 0.0], [1.0, 0.01]; channel_map=[1, 2])
set_ts!(source, 2pi * 220)
start_DESource(source, get_default_output_device(); sample_rate=48000,
    buffer_size=256, substeps=4)
set_param!(source, 2, 0.02)
set_gain!(source, 0.5)
stop_DESource(source)
```

`get_devices()` lists device names, channel counts, and indices. Select an
output-capable device explicitly if the system default is unsuitable.

An SDE supplies an additional in-place noise function:

```julia
noise!(du, u, p, t) = fill!(du, p[3])
source = DESource(oscillator!, noise!, [0.1, 0.0], [1.0, 0.01, 0.001])
```

## Solver and audio behavior

The `alg` keyword selects a SciML algorithm. Defaults are `Tsit5()` for ODEs and
`SOSRA()` for SDEs. A compatible stiff ODE algorithm can also be passed after
loading its SciML package; allocation and timing depend on that algorithm and
the user equation. This branch uses fixed steps, with
`dt = timescale / sample_rate / substeps`. It does not enable adaptive error
control or promise compatibility with every algorithm.

The producer reuses its integrator, sample buffer, and typed rendering loop.
PortAudio's callback copies completed samples from a bounded queue. The
default four queued blocks provide scheduling headroom and add latency.

Channel maps accept variable indices (`0` means silence), lists of indices,
or a variables-by-channels gain matrix. Controls publish copies and take effect
at block boundaries. Getters return copies: use `set_param!`, `set_gain!`,
`set_ts!`, and `set_channelmap!` to change playback. Stop playback before
changing its channel count.

`stop_DESource` preserves the last produced solver state. Starting again
continues from it; `reset_state!` requests the initial state. Samples already
queued may be ahead of the last sample heard. Non-finite output or solver
failure silences the block and latches silence until reset or preparation.

## Offline rendering and tests

```julia
prepare!(source; sample_rate=48000, buffer_size=256, substeps=4)
buffer = zeros(Float32, 512) # 256 interleaved stereo frames
render!(buffer, source)
```

Repeated calls continue the same trajectory. Offline rendering and inspection
require a stopped source. `integrator_info(source)` and
`audio_debug_status(source)` expose state and stream counters.

Run the suite from the package checkout:

```sh
julia --threads=2 --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Tests cover numerical sample timing, block continuity, control changes, reset,
SDE variance and noise storage, queue wraparound, concurrent publication,
failure recovery, and operation beyond one million solver steps. The warmed
tested ODE/SDE rendering and callback paths allocate zero Julia heap bytes.
Setup, control publication, resets, other algorithms, and allocating user
equations are outside that claim. Julia GC and operating-system scheduling
still prevent a hard real-time guarantee.
