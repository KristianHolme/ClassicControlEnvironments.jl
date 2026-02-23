using ClassicControlEnvironments
using Drill
using WGLMakie
using Zygote
## setup env, alg, policy and agent
alg = PPO(; n_steps = 128, batch_size = 128, learning_rate = 2.343f-4, epochs = 30, ent_coef = 0.00981f0, vf_coef = 0.66482f0, gamma = 0.98846f0, gae_lambda = 0.75575f0, clip_range = 0.24175f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() for _ in 1:16])
pendenv = MonitorWrapperEnv(pendenv)
pendenv = NormalizeWrapperEnv(pendenv, gamma = alg.gamma)

pendpolicy = ActorCriticLayer(observation_space(pendenv), action_space(pendenv))
pendagent = Agent(pendpolicy, alg; verbose = 2)
## train agent
train!(pendagent, pendenv, alg, 100_000)
## collect trajectory
single_env = PendulumEnv()
obs, actions, rewards = collect_trajectory(pendagent, single_env; norm_env = pendenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
fig
