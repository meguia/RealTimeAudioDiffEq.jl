# RealTimeAudioDiffEq

Audio from ordinary and stochastic differential equations (ODEs and SDEs),
with experimental support for ordinary and neutral delay equations (DDEs).

This branch, `feature/sciml-delay-audio`, builds on
`feature/integrator-interface-audio` and adds a single DDE path using
DelayDiffEq. Both keep runtime code in `src/rt_DE.jl`.

**Status:** the ODE/SDE renderer retains its tested zero-allocation paths.
The DDE renderer uses SciML's actual interpolation and adaptive solver, but
allocates while recording history. 

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
the user equation. ODE/SDE rendering uses fixed steps, with
`dt = timescale / sample_rate / substeps`. ODE/SDE rendering does not enable adaptive
error control. DDE rendering does; neither path promises compatibility with
every algorithm.

The producer reuses its integrator, sample buffer, and typed rendering loop.
PortAudio's callback copies completed samples from a bounded queue. The
default four queued blocks provide scheduling headroom and add latency.
Runtime code remains in `src/rt_DE.jl`; the PortAudio bindings remain in
`src/libportaudio.jl`.

Channel maps accept variable indices (`0` means silence), lists of indices,
or a variables-by-channels gain matrix. Controls publish copies and take effect
at block boundaries. Getters return copies: use `set_param!`, `set_gain!`,
`set_ts!`, and `set_channelmap!` to change playback. Stop playback before
changing its channel count.

`stop_DESource` preserves the last produced solver state. Starting again
continues from it; `reset_state!` requests the initial state. Samples already
queued may be ahead of the last sample heard. Non-finite output or solver
failure silences the block and latches silence until reset or preparation.

## Delay equations and native interpolation

An ordinary DDE reads a past state, `h(p, t-delay)`. A neutral DDE also reads
its past derivative, `h(p, t-delay, Val{1})`. Use `neutral=true` only when the
model actually needs delayed derivatives. Both use the same constructor and
rendering code.

```julia
using SciMLBase: DDEProblem
using DelayDiffEq: MethodOfSteps
using OrdinaryDiffEqTsit5: Tsit5

history(p, t; idxs=nothing) = idxs === nothing ? [0.1] : 0.1
history(p, t, ::Type{Val{1}}; idxs=nothing) = idxs === nothing ? [0.0] : 0.0

function delayed_feedback!(du, u, h, p, t)
    du[1] = -p[1]*u[1] + p[2]*h(p, t-p[3], Val{1}; idxs=1)
end

p = [1.0, 0.2, 0.1]
problem = DDEProblem(delayed_feedback!, [0.1], history, (0.0, 10.0), p;
    constant_lags=(p[3],), neutral=true)
source = DESource(problem; alg=MethodOfSteps(Tsit5()), max_delay=p[3],
    channel_map=[1, 1], abstol=1e-12, reltol=1e-10,
    history_capacity=100_000)
prepare!(source; sample_rate=48000, buffer_size=256)
buffer = zeros(Float32, 512)
render!(buffer, source)
```

DelayDiffEq owns stepping, error control, discontinuity propagation, and
interpolation. A delayed derivative is the derivative of its continuous
interpolant, not a finite difference between audio samples. There are no
handwritten Tsit5 interpolation coefficients in this package. For a stiff
problem, a compatible method such as `MethodOfSteps(Rodas5P())` can be passed
after importing `Rodas5P` from OrdinaryDiffEqRosenbrock. The tests exercise this
method on a stiff delayed equation as well as `Tsit5` on a non-stiff one.

The audio grid remains regular. Adaptive internal steps end at each grid
point and are limited to `timescale / sample_rate / substeps`. This interval
must not exceed the smallest declared lag. `abstol` and `reltol` control local
solver error, not a global accuracy guarantee or a deadline. A low-order
method with very tight tolerances can require too many steps for audio.

### History storage and current limits

At the end of each block, a small adapter removes obsolete knots from
DelayDiffEq's saved history, retaining `max_delay` plus boundary and rollback
knots. SciML continues to evaluate the retained interpolant. The adapter
accesses solver bookkeeping, so this branch pins DelayDiffEq 6.3.1 and
OrdinaryDiffEqCore 4.19.0. Changing either requires retesting the adapter.
This dependency is the compromise needed for a rolling window: simply
turning off solution saving does not stop DelayDiffEq saving its history.

The number of stored knots also has an explicit `history_capacity` limit.
If the history needed within the delay window reaches that limit, rendering
silences the block and latches failure. It does not discard needed history
to keep playing. This bounds retained knot count, not total allocated bytes
or work per block. SciML still allocates new history entries, and trimming
arrays moves references; GC and occasional deadline misses remain concerns.

The supported scope is in-place equations with `Vector{Float64}` states and
parameters, identity mass matrix, and declared positive constant lags.
Neutral equations currently support one lag to bound recurring discontinuity
queues. Ordinary DDEs may declare multiple lags. State-dependent delays,
callbacks, stochastic DDEs, and queries beyond the declared history window
are outside this implementation. Supply derivative prehistory for a neutral
equation, including any incompatibility at the initial time.

Stopping and preparing again retains the complete DDE solver and its history.
Changing parameters requires `reset_state!`. Changing timescale or step size
requires stopping, resetting, and preparing again. A reset starts a new
trajectory using the supplied prehistory. Changes to delay values require a
new problem with matching `constant_lags`. Gain and channel maps can change
without restarting the trajectory. Continuous parameter automation is not
yet provided for this DDE path.

For offline derived outputs, `render!(buffer, source; observe=f)` calls
`f(u,p,t,h)` at each sample and uses its returned channel values instead of
the channel map. For DDEs, `h` is the actual SciML history function. Playback
through `start_DESource` currently uses the channel map.

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
