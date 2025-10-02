@testitem "MountainCar discrete observation range" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    reset!(env)

    # Test observations after initialization
    obs = observe(env)
    @test length(obs) == 2
    @test env.problem.min_position <= obs[1] <= env.problem.max_position  # position
    @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed      # velocity

    # Test observations after actions
    rng = MersenneTwister(123)
    for _ in 1:20
        action = rand(rng, 0:2)  # Random discrete action
        act!(env, action)
        local obs = observe(env)
        @test env.problem.min_position <= obs[1] <= env.problem.max_position
        @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed
    end
end

@testitem "MountainCar continuous observation range" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    reset!(env)

    # Test observations after initialization
    obs = observe(env)
    @test length(obs) == 2
    @test env.problem.min_position <= obs[1] <= env.problem.max_position  # position
    @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed      # velocity

    # Test observations after actions
    rng = MersenneTwister(123)
    for _ in 1:20
        action = rand(rng, Float32) * 2.0f0 - 1.0f0  # Random action in [-1, 1]
        act!(env, action)
        local obs = observe(env)
        @test env.problem.min_position <= obs[1] <= env.problem.max_position
        @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed
    end
end

@testitem "MountainCar discrete initial state" begin
    using DRiL
    using Random
    rng = MersenneTwister(456)
    env = MountainCarEnv(rng = rng)

    # Test multiple resets
    for _ in 1:10
        reset!(env)
        @test -0.6f0 <= env.problem.position <= -0.4f0  # Initial position range
        @test env.problem.velocity == 0.0f0             # Initial velocity
        @test env.step == 0                             # Step counter reset
        @test !terminated(env)                          # Not terminated initially
        @test !truncated(env)                           # Not truncated initially
    end
end

@testitem "MountainCar continuous initial state" begin
    using DRiL
    using Random
    rng = MersenneTwister(456)
    env = MountainCarContinuousEnv(rng = rng)

    # Test multiple resets
    for _ in 1:10
        reset!(env)
        @test -0.6f0 <= env.problem.position <= -0.4f0  # Initial position range
        @test env.problem.velocity == 0.0f0             # Initial velocity
        @test env.step == 0                             # Step counter reset
        @test !terminated(env)                          # Not terminated initially
        @test !truncated(env)                           # Not truncated initially
    end
end

@testitem "MountainCar discrete physics simulation" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    reset!(env)

    initial_position = env.problem.position
    initial_velocity = env.problem.velocity

    # Apply action
    act!(env, 2)  # Push right
    @test env.problem.force == 1.0f0  # Should map to +1 force

    # Position should change based on velocity
    # Velocity should be affected by force and gravity
    @test env.problem.position != initial_position || env.problem.velocity != initial_velocity

    # Test boundary conditions
    env.problem.position = env.problem.min_position - 0.1f0
    env.problem.velocity = -0.01f0
    act!(env, 0)  # Push left
    @test env.problem.position == env.problem.min_position  # Should be clamped
    @test env.problem.velocity == 0.0f0  # Velocity should be zeroed at boundary
end

@testitem "MountainCar continuous physics simulation" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    reset!(env)

    initial_position = env.problem.position
    initial_velocity = env.problem.velocity

    # Apply action
    act!(env, 1.0f0)  # Push right
    @test env.problem.force == 1.0f0

    # Position should change based on velocity
    # Velocity should be affected by force and gravity
    @test env.problem.position != initial_position || env.problem.velocity != initial_velocity

    # Test boundary conditions
    env.problem.position = env.problem.min_position - 0.1f0
    env.problem.velocity = -0.01f0
    act!(env, -1.0f0)  # Push left
    @test env.problem.position == env.problem.min_position  # Should be clamped
    @test env.problem.velocity == 0.0f0  # Velocity should be zeroed at boundary
end

@testitem "MountainCar discrete goal condition" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    reset!(env)

    # Manually set car to goal position with sufficient velocity
    env.problem.position = env.problem.goal_position
    env.problem.velocity = env.problem.goal_velocity

    @test terminated(env)

    # Test goal reward
    reward_val = ClassicControlEnvironments.reward(env)
    @test reward_val ≈ -1.0f0  # Always negative reward

    # Test not at goal
    env.problem.position = 0.0f0
    @test !terminated(env)
end

@testitem "MountainCar continuous goal condition" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    reset!(env)

    # Manually set car to goal position with sufficient velocity
    env.problem.position = env.problem.goal_position
    env.problem.velocity = env.problem.goal_velocity

    @test terminated(env)

    # Test goal reward
    reward_val = ClassicControlEnvironments.reward(env)
    @test reward_val > 0  # Should get positive reward at goal

    # Test not at goal
    env.problem.position = 0.0f0
    @test !terminated(env)
end

@testitem "MountainCar discrete reward structure" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    reset!(env)

    # Test step penalty with no force
    initial_reward = act!(env, 1)  # No force (action 1)
    @test initial_reward < 0  # Should get negative reward or zero
end

@testitem "MountainCar continuous reward structure" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    reset!(env)

    # Test step penalty with no force
    initial_reward = act!(env, 0.0f0)  # No force applied
    @test initial_reward ≤ 0  # Should get negative reward or zero
end

@testitem "MountainCar discrete step count and truncation" begin
    using DRiL
    using Random
    max_steps = 50
    env = MountainCarEnv(max_steps = max_steps)
    reset!(env)

    @test env.step == 0
    @test !truncated(env)

    for i in 1:(max_steps - 1)
        act!(env, 1)  # No force
        @test env.step == i
        @test !truncated(env)
    end

    act!(env, 1)  # No force
    @test env.step == max_steps
    @test truncated(env)
end

@testitem "MountainCar continuous step count and truncation" begin
    using DRiL
    using Random
    max_steps = 50
    env = MountainCarContinuousEnv(max_steps = max_steps)
    reset!(env)

    @test env.step == 0
    @test !truncated(env)

    for i in 1:(max_steps - 1)
        act!(env, 0.0f0)  # No force
        @test env.step == i
        @test !truncated(env)
    end

    act!(env, 0.0f0)  # No force
    @test env.step == max_steps
    @test truncated(env)
end

@testitem "MountainCar discrete action space" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    as = action_space(env)
    @test as isa Discrete
    @test as.n == 3
    @test as.start == 0
    @test 0 ∈ as
    @test 1 ∈ as
    @test 2 ∈ as
    @test !(3 ∈ as)
end

@testitem "MountainCar discrete action mapping" begin
    using DRiL
    using Random
    env = MountainCarEnv()

    reset!(env)
    # Test left action (0 -> -1 force)
    act!(env, 0)
    @test env.problem.force == -1.0f0

    reset!(env)
    # Test no action (1 -> 0 force)
    act!(env, 1)
    @test env.problem.force == 0.0f0

    reset!(env)
    # Test right action (2 -> +1 force)
    act!(env, 2)
    @test env.problem.force == 1.0f0
end

@testitem "MountainCar discrete array action input" begin
    using DRiL
    using Random
    env = MountainCarEnv()
    reset!(env)
    act!(env, [2])  # Should work with array input
    @test env.problem.force == 1.0f0
end

@testitem "MountainCar continuous action space" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    as = action_space(env)
    @test as isa Box{Float32}
    @test as.low == [-1.0f0]
    @test as.high == [1.0f0]
end

@testitem "MountainCar continuous action clamping" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()

    reset!(env)
    # Test that extreme actions are clamped
    act!(env, 10.0f0)  # Should be clamped to 1.0
    @test env.problem.force == 1.0f0

    reset!(env)
    act!(env, -10.0f0)  # Should be clamped to -1.0
    @test env.problem.force == -1.0f0
end

@testitem "MountainCar continuous array action input" begin
    using DRiL
    using Random
    env = MountainCarContinuousEnv()
    reset!(env)
    act!(env, [0.5f0])  # Should work with array input
    @test env.problem.force == 0.5f0
end

@testitem "MountainCar common observation space" begin
    using DRiL
    using Random
    discrete_env = MountainCarEnv()
    continuous_env = MountainCarContinuousEnv()

    # Both should have identical observation spaces
    os_discrete = observation_space(discrete_env)
    os_continuous = observation_space(continuous_env)

    @test os_discrete.low == os_continuous.low
    @test os_discrete.high == os_continuous.high
    @test os_discrete.shape == os_continuous.shape

    @test os_discrete.low[1] == discrete_env.problem.min_position
    @test os_discrete.high[1] == discrete_env.problem.max_position
    @test os_discrete.low[2] == -discrete_env.problem.max_speed
    @test os_discrete.high[2] == discrete_env.problem.max_speed
end

@testitem "MountainCar type hierarchy" begin
    using DRiL
    using Random
    discrete_env = MountainCarEnv()
    continuous_env = MountainCarContinuousEnv()

    @test discrete_env isa AbstractMountainCarEnv
    @test continuous_env isa AbstractMountainCarEnv
    @test discrete_env isa AbstractEnv
    @test continuous_env isa AbstractEnv
end
