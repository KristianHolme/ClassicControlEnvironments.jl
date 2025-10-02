@testitem "Acrobot basic interface compliance" begin
    using DRiL
    using Random
    env = AcrobotEnv()

    # Test that all required methods exist
    @test hasmethod(DRiL.reset!, (typeof(env),))
    @test hasmethod(DRiL.act!, (typeof(env), Integer))
    @test hasmethod(DRiL.act!, (typeof(env), AbstractArray{Int, 1}))
    @test hasmethod(DRiL.observe, (typeof(env),))
    @test hasmethod(DRiL.terminated, (typeof(env),))
    @test hasmethod(DRiL.truncated, (typeof(env),))
    @test hasmethod(DRiL.action_space, (typeof(env),))
    @test hasmethod(DRiL.observation_space, (typeof(env),))
    @test hasmethod(DRiL.get_info, (typeof(env),))

    # Test environment type hierarchy
    @test env isa AbstractEnv
end

@testitem "Acrobot action space" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    act_space = action_space(env)

    @test act_space isa Discrete
    @test act_space.n == 3
    @test act_space.start == 0
    @test 0 ∈ act_space
    @test 1 ∈ act_space
    @test 2 ∈ act_space
    @test !(3 ∈ act_space)
    @test !(-1 ∈ act_space)
end

@testitem "Acrobot observation space" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    os = observation_space(env)

    @test os isa Box{Float32}
    @test length(os.low) == 6
    @test length(os.high) == 6

    # Check observation bounds
    @test os.low[1] == -1.0f0  # cos(θ₁)
    @test os.high[1] == 1.0f0
    @test os.low[2] == -1.0f0  # sin(θ₁)
    @test os.high[2] == 1.0f0
    @test os.low[3] == -1.0f0  # cos(θ₂)
    @test os.high[3] == 1.0f0
    @test os.low[4] == -1.0f0  # sin(θ₂)
    @test os.high[4] == 1.0f0
    @test os.low[5] == -env.problem.max_vel_1  # θ̇₁
    @test os.high[5] == env.problem.max_vel_1
    @test os.low[6] == -env.problem.max_vel_2  # θ̇₂
    @test os.high[6] == env.problem.max_vel_2
end

@testitem "Acrobot initial state and reset" begin
    using DRiL
    using Random
    rng = MersenneTwister(123)
    env = AcrobotEnv(rng = rng)

    # Test multiple resets
    for _ in 1:10
        reset!(env)

        # Check that step counter is reset
        @test env.step == 0
        @test !terminated(env)
        @test !truncated(env)

        # Check that state is within initialization bounds
        p = env.problem
        @test -0.1f0 <= p.theta1 <= 0.1f0
        @test -0.1f0 <= p.theta2 <= 0.1f0
        @test -0.1f0 <= p.dtheta1 <= 0.1f0
        @test -0.1f0 <= p.dtheta2 <= 0.1f0
        @test p.torque == 0.0f0
    end
end

@testitem "Acrobot observation consistency" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Test that observations are within bounds
    for _ in 1:50
        obs = observe(env)
        @test length(obs) == 6
        @test obs ∈ observation_space(env)

        # Test trigonometric consistency
        cos_theta1, sin_theta1 = obs[1], obs[2]
        cos_theta2, sin_theta2 = obs[3], obs[4]

        # cos²(θ) + sin²(θ) = 1 (with small tolerance for numerical errors)
        @test abs(cos_theta1^2 + sin_theta1^2 - 1.0f0) < 1.0e-5
        @test abs(cos_theta2^2 + sin_theta2^2 - 1.0f0) < 1.0e-5

        # Velocities should be within bounds
        @test abs(obs[5]) <= env.problem.max_vel_1
        @test abs(obs[6]) <= env.problem.max_vel_2

        # Take random action and continue
        action = rand(0:2)
        act!(env, action)

        if terminated(env) || truncated(env)
            break
        end
    end
end

@testitem "Acrobot action mapping" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Test action 0 (torque -1)
    act!(env, 0)
    @test env.problem.torque == -1.0f0

    reset!(env)
    # Test action 1 (torque 0)
    act!(env, 1)
    @test env.problem.torque == 0.0f0

    reset!(env)
    # Test action 2 (torque +1)
    act!(env, 2)
    @test env.problem.torque == 1.0f0
end

@testitem "Acrobot array action input" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Test that array actions work
    act!(env, [2])
    @test env.problem.torque == 1.0f0

    reset!(env)
    act!(env, [0])
    @test env.problem.torque == -1.0f0
end

@testitem "Acrobot physics simulation" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Store initial state
    initial_theta1 = env.problem.theta1
    initial_theta2 = env.problem.theta2
    initial_dtheta1 = env.problem.dtheta1
    initial_dtheta2 = env.problem.dtheta2

    # Apply torque and check that state changes
    act!(env, 2)  # Apply positive torque

    # At least one state variable should have changed
    @test (
        env.problem.theta1 != initial_theta1 ||
            env.problem.theta2 != initial_theta2 ||
            env.problem.dtheta1 != initial_dtheta1 ||
            env.problem.dtheta2 != initial_dtheta2
    )
end

@testitem "Acrobot angle wrapping" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Set angles to values that should wrap
    env.problem.theta1 = 4.0f0 * π
    env.problem.theta2 = -4.0f0 * π

    # Apply action to trigger angle wrapping
    act!(env, 1)

    # Angles should be wrapped to [-π, π]
    @test -π <= env.problem.theta1 <= π
    @test -π <= env.problem.theta2 <= π
end

@testitem "Acrobot velocity bounds" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Set velocities beyond bounds
    env.problem.dtheta1 = 2.0f0 * env.problem.max_vel_1
    env.problem.dtheta2 = 2.0f0 * env.problem.max_vel_2

    # Apply action to trigger velocity bounding
    act!(env, 1)

    # Velocities should be bounded
    @test abs(env.problem.dtheta1) <= env.problem.max_vel_1
    @test abs(env.problem.dtheta2) <= env.problem.max_vel_2
end

@testitem "Acrobot reward structure" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Before reaching goal, reward should be -1
    reward_val = act!(env, 1)
    @test reward_val == -1.0f0

    # Manually set goal condition
    env.problem.theta1 = 0.0f0
    env.problem.theta2 = 0.0f0
    # This gives -cos(0) - cos(0) = -2, which is not > 1
    @test !terminated(env)

    # Set to goal condition: -cos(θ₁) - cos(θ₂ + θ₁) > 1.0
    # One way: θ₁ = π, θ₂ = 0 gives -cos(π) - cos(π) = 1 + 1 = 2 > 1
    env.problem.theta1 = π
    env.problem.theta2 = 0.0f0
    @test terminated(env)

    # Reward when terminated should be 0
    env.step = 0  # Reset step counter to ensure we're not truncated
    @test ClassicControlEnvironments.reward(env) == 0.0f0
end

@testitem "Acrobot goal condition" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    # Test various goal conditions
    test_cases = [
        (π, 0.0f0, true),      # θ₁=π, θ₂=0: -(-1) - (-1) = 2 > 1 ✓
        (0.0f0, π, false),      # θ₁=0, θ₂=π: -1 - (-1) = 0 < 1 ✗
        (π / 2, π / 2, false),      # θ₁=π/2, θ₂=π/2: -0 - (-1) = 1 = 1 ✗
        (2π / 3, π / 4, true),    # θ₁=2π/3, θ₂=-π/4: -cos(2π/3) - cos(2π/3 - π/4) \isapprox 1.46 < 1 ✗
    ]

    for (theta1, theta2, should_terminate) in test_cases
        env.problem.theta1 = theta1
        env.problem.theta2 = theta2

        height = -cos(theta1) - cos(theta2 + theta1)
        expected = height > 1.0f0
        # @info "theta1 = $theta1, theta2 = $theta2, should_terminate = $should_terminate, expected = $expected"
        # @info "height = $height"
        @test should_terminate == expected
        @test terminated(env) == expected
    end
end

@testitem "Acrobot step count and truncation" begin
    using DRiL
    using Random
    max_steps = 50
    env = AcrobotEnv(max_steps = max_steps)
    reset!(env)

    @test env.step == 0
    @test !truncated(env)

    # Run for max_steps-1
    for i in 1:(max_steps - 1)
        act!(env, 1)  # No torque
        @test env.step == i
        @test !truncated(env)
    end

    # One more step should trigger truncation
    act!(env, 1)
    @test env.step == max_steps
    @test truncated(env)
end

@testitem "Acrobot dynamics variants" begin
    using DRiL
    using Random
    # Test both "book" and "nips" dynamics
    env_book = AcrobotEnv(book_or_nips = "book")
    env_nips = AcrobotEnv(book_or_nips = "nips")

    @test env_book.problem.book_or_nips == "book"
    @test env_nips.problem.book_or_nips == "nips"

    # Both should work without errors
    reset!(env_book)
    reset!(env_nips)

    for _ in 1:10
        act!(env_book, rand(0:2))
        act!(env_nips, rand(0:2))
    end
end

@testitem "Acrobot problem parameters" begin
    using DRiL
    using Random
    # Test custom parameters
    custom_env = AcrobotEnv(
        link_length_1 = 2.0f0,
        link_mass_1 = 2.0f0,
        max_vel_1 = 8.0f0 * π,
        dt = 0.1f0
    )

    @test custom_env.problem.link_length_1 == 2.0f0
    @test custom_env.problem.link_mass_1 == 2.0f0
    @test custom_env.problem.max_vel_1 == 8.0f0 * π
    @test custom_env.problem.dt == 0.1f0

    # Observation space should reflect the custom velocity bounds
    os = observation_space(custom_env)
    @test os.low[5] == -8.0f0 * π
    @test os.high[5] == 8.0f0 * π
end

@testitem "Acrobot get info" begin
    using DRiL
    using Random
    env = AcrobotEnv()
    reset!(env)

    info = get_info(env)
    @test info isa Dict{String, Any}
    @test haskey(info, "step")
    @test haskey(info, "theta1")
    @test haskey(info, "theta2")
    @test haskey(info, "dtheta1")
    @test haskey(info, "dtheta2")

    @test info["step"] == env.step
    @test info["theta1"] == env.problem.theta1
    @test info["theta2"] == env.problem.theta2
    @test info["dtheta1"] == env.problem.dtheta1
    @test info["dtheta2"] == env.problem.dtheta2
end

@testitem "Acrobot max step limit prevents infinite loops" begin
    using DRiL
    using Random
    # Test with a very small max_steps to ensure we hit the limit
    max_steps = 10
    env = AcrobotEnv(max_steps = max_steps)
    reset!(env)

    # Manually ensure we won't reach goal (set to initial position)
    env.problem.theta1 = 0.0f0
    env.problem.theta2 = 0.0f0
    env.problem.dtheta1 = 0.0f0
    env.problem.dtheta2 = 0.0f0

    let step_count = 0
        max_iterations = max_steps + 5  # Safety buffer

        while !terminated(env) && !truncated(env) && step_count < max_iterations
            act!(env, 1)  # Apply no torque (action 1)
            step_count += 1

            # Verify step counter is incrementing
            @test env.step == step_count

            # Safety check to prevent actual infinite loop in test
            if step_count >= max_iterations
                error("Test exceeded maximum iterations - potential infinite loop!")
            end
        end

        # Should be truncated, not terminated (since we set it to not reach goal)
        @test truncated(env)
        @test !terminated(env)
        @test env.step == max_steps
        @test step_count == max_steps
    end
end

@testitem "Acrobot episode termination conditions" begin
    using DRiL
    using Random
    # Test that episodes can end either by goal or max steps
    max_steps = 20
    env = AcrobotEnv(max_steps = max_steps)

    # Test 1: Goal termination
    reset!(env)
    # Manually set goal condition: -cos(θ₁) - cos(θ₂ + θ₁) > 1.0
    env.problem.theta1 = π
    env.problem.theta2 = 0.0f0
    env.step = 5  # Set step to some value < max_steps

    @test terminated(env)
    @test !truncated(env)
    @test ClassicControlEnvironments.reward(env) == 0.0f0

    # Test 2: Max steps truncation (already tested above but adding here for completeness)
    reset!(env)
    env.step = max_steps

    @test !terminated(env)  # Not at goal
    @test truncated(env)    # But truncated due to max steps
end

@testitem "Acrobot long episode simulation" begin
    using DRiL
    using Random
    env = AcrobotEnv(max_steps = 100)
    reset!(env)

    let ep_length = 0, tot_reward = 0.0f0
        while !terminated(env) && !truncated(env)
            actn = rand(0:2)
            rew = act!(env, actn)
            tot_reward += rew
            ep_length += 1

            # Ensure observations remain valid
            obs = observe(env)
            @test obs ∈ observation_space(env)

            # Safety check to prevent infinite loop in tests
            if ep_length > 200
                error("Episode exceeded reasonable length - potential infinite loop!")
            end
        end

        @test ep_length > 0
        @test tot_reward <= 0.0f0  # All rewards are -1 or 0

        if terminated(env)
            @test tot_reward == -(ep_length - 1)  # Last reward is 0
        else
            @test truncated(env)
            @test tot_reward == -ep_length  # All rewards are -1
        end
    end
end
