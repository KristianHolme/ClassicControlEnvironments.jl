# ClassicControlEnvironments

[![Build Status](https://github.com/KristianHolme/Pendulum.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/Pendulum.jl/actions/workflows/CI.yml?query=branch%3Amain)

Classic control environments for Reinforcement Learning. All environments implement the DrillInterface API (from [Drill.jl](https://github.com/KristianHolme/Drill.jl)) and work with Drill.jl for training.

## Interface

Environments follow the API used by [Drill.jl](https://github.com/KristianHolme/Drill.jl), from the [DrillInterface.jl](https://github.com/KristianHolme/Drill.jl/tree/main/DrillInterface)

## Environments

| Environment                  | Type       | Description                               |
| ---------------------------- | ---------- | ----------------------------------------- |
| **CartPoleEnv**              | Discrete   | Inverted pendulum on a cart (2 actions).  |
| **MountainCarEnv**           | Discrete   | Car in a valley (3 actions).              |
| **MountainCarContinuousEnv** | Continuous | Same task with continuous force.          |
| **AcrobotEnv**               | Discrete   | Two-link acrobot (3 actions).             |
| **PendulumEnv**              | Continuous | Inverted pendulum with continuous torque. |

Constructors accept optional `problem`, `max_steps`, `rng`, and `kwargs` for the underlying problem (e.g. `CartPoleEnv(; max_steps = 500)`).

## Installation

```julia
] add ClassicControlEnvironments
```

For plotting, add Makie and a backend (e.g. `Makie`, `WGLMakie`). For training, add [Drill.jl](https://github.com/KristianHolme/Drill.jl).

## Quick start

### Create an environment

```julia
using ClassicControlEnvironments

env = CartPoleEnv()
# or: env = PendulumEnv(; max_steps = 200)
```

All envs implement the DrillInterface API, so you can call `reset!(env)`, `act!(env, action)`, `observe(env)`, `terminated(env)`, `truncated(env)`, `action_space(env)`, `observation_space(env)`, etc.

### Plotting (Makie)

Load a Makie backend (e.g. `using WGLMakie`) so the optional Makie extension is used.

- **Current state**: `plot(env.problem)` plots the current state (e.g. pendulum angle, cart position).
- **Trajectory (after a rollout)**: `plot_trajectory(env, obs, actions, rewards)` gives a static figure; `plot_trajectory_interactive(env, obs, actions, rewards)` gives an interactive figure with a step slider. The vectors `obs`, `actions`, `rewards` typically come from `collect_trajectory` in Drill.
- **Manual control / interaction**: `interactive_viz(env)` opens a live visualization where you control the environment with the keyboard (e.g. arrow keys) or on-screen buttons: step with actions, reset, and optionally run with random or constant actions. Use this to try an environment by hand. For programmatic live updates (e.g. in a custom loop), `live_viz(env.problem)` returns a figure and an `update_viz!` callback to refresh the view after stepping.

### Training with Drill.jl

```julia
using ClassicControlEnvironments
using Drill
using WGLMakie  # for plotting after training
using Zygote    # for PPO

alg = PPO(; n_steps = 128, batch_size = 128, learning_rate = 3.0f-4, epochs = 10)
env = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:8])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma = alg.gamma)

policy = ActorCriticLayer(observation_space(env), action_space(env))
agent = Agent(policy, alg; verbose = 2)
train!(agent, env, alg, 100_000)

# Collect a trajectory and plot
single_env = CartPoleEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env = env)
fig = plot_trajectory(single_env, obs, actions, rewards)
# or: fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
```

Full examples (CartPole, MountainCar, Acrobot, Pendulum) are in the `examples/` directory.
