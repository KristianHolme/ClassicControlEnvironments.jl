module ClassicControlEnvironments

using DrillInterface
using Random
using Reexport

include("utils.jl")
# Export the main environments
include("Pendulum.jl")
export PendulumEnv, PendulumProblem

include("MountainCar.jl")
export AbstractMountainCarEnv, MountainCarContinuousEnv, MountainCarEnv, MountainCarProblem

include("CartPole.jl")
export CartPoleEnv, CartPoleProblem

include("Acrobot.jl")
export AcrobotEnv, AcrobotProblem

function plot end
function live_viz end
function interactive_viz end
function plot_trajectory end
function plot_trajectory_interactive end
function animate_trajectory_video end
function plot_trajectory_phase_space end
export animate_trajectory_video, interactive_viz, live_viz, plot, plot_trajectory,
    plot_trajectory_interactive

end
