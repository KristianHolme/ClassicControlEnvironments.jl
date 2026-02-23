using ClassicControlEnvironments
using Drill
using WGLMakie
using Zygote
## setup env, alg, policy and agent
alg = PPO(; n_steps = 128, batch_size = 128, learning_rate = 3.0f-4, epochs = 10)
acrobotenv = BroadcastedParallelEnv([AcrobotEnv() for _ in 1:8])
acrobotenv = MonitorWrapperEnv(acrobotenv)
acrobotenv = NormalizeWrapperEnv(acrobotenv, gamma = alg.gamma)

acrobotpolicy = ActorCriticLayer(observation_space(acrobotenv), action_space(acrobotenv))
acrobotagent = Agent(acrobotpolicy, alg; verbose = 2)
## train agent
train!(acrobotagent, acrobotenv, alg, 100_000)
## collect trajectory
single_env = AcrobotEnv()
obs, actions, rewards = collect_trajectory(acrobotagent, single_env; norm_env = acrobotenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
fig, _ = plot_trajectory_interactive(single_env, obs, actions, rewards)
fig
