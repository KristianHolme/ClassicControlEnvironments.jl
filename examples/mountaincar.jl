using ClassicControlEnvironments
using DRiL
using WGLMakie
using Zygote
## setup env, alg, policy and agent
alg = PPO(; ent_coef = 0.01f0, n_steps = 256, batch_size = 64, epochs = 10)
env = BroadcastedParallelEnv([MountainCarEnv() for _ in 1:16])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma = alg.gamma)

policy = ActorCriticLayer(observation_space(env), action_space(env))
agent = Agent(policy, alg; verbose = 2)
## train agent
train!(agent, env, alg, 100_000)
## collect trajectory
single_env = MountainCarEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env = env)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
fig
