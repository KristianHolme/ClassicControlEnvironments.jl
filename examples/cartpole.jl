using ClassicControlEnvironments
using DRiL
using WGLMakie
using Zygote
## setup env, alg, policy and agent
alg = PPO(; n_steps = 128, batch_size = 128, learning_rate = 3.0f-4, epochs = 10)
cartpoleenv = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:8])
cartpoleenv = MonitorWrapperEnv(cartpoleenv)
cartpoleenv = NormalizeWrapperEnv(cartpoleenv, gamma = alg.gamma)

cartpolepolicy = ActorCriticLayer(observation_space(cartpoleenv), action_space(cartpoleenv))
cartpoleagent = Agent(cartpolepolicy, alg; verbose = 2)
## train agent
train!(cartpoleagent, cartpoleenv, alg, 100_000)
## collect trajectory
single_env = CartPoleEnv()
obs, actions, rewards = collect_trajectory(cartpoleagent, single_env; norm_env = cartpoleenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
fig
