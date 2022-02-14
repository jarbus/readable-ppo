using Flux
using Flux.Optimise: update!
using Plots
using StableRNGs

include("CartPole.jl")

env = CartPoleEnv()
rng = StableRNG(123)

sample(x::Vector{Float32}) = rand() < x[1] ? 1 : 2
mean(x) = sum(x)/length(x)

γ = 0.99
ϵ = 0.2

# First vector is timestep, second vector is num_workers
Base.@kwdef mutable struct PPOTrajectory
    states    ::Vector{Vector{Vector{Float32}}} = []
    logprobs  ::Vector{Vector{Vector{Float32}}} = []
    actions   ::Vector{Vector{Int32}}           = []
    values    ::Vector{Vector{Float32}}         = []
    rewards   ::Vector{Vector{Float32}}         = []
    dones     ::Vector{Vector{Float32}}         = []
    advantages::Vector{Vector{Float32}}         = []
end


function compute_generalized_advantage_estimates!(traj::PPOTrajectory; γ=0.99, λ=0.95)
    @assert length(traj.values) > 1
    # length(traj.values) = T+1 bc we need vₜ₊₁)
    T = length(traj.values) - 1
    Aₜ₊₁ = 0
    for t in T:-1:1
        # ignore t+1 if t is end of episode and t+1 is start of next
        mask = 1.0 .- traj.dones[t]
        Aₜ₊₁ = Aₜ₊₁ .* mask
        vₜ₊₁ = traj.values[t+1] .* mask

        δₜ = traj.rewards[t] .+ γ*vₜ₊₁ .- traj.values[t]
        Aₜ = δₜ .+ (λ*γ*Aₜ₊₁)
        pushfirst!(traj.advantages, Aₜ)

        Aₜ₊₁ = Aₜ
    end
end

to_matrix(x::Vector{Vector{Vector{Float32}}}) = hcat(vcat(x...)...)
to_matrix(x::Vector{Vector{Int32}}) = hcat(vcat(x...))
to_matrix(x::Vector{Vector{Float32}}) = hcat(vcat(x...))

function update_weights!(opt, actor, critic, traj::PPOTrajectory; vf_coeff=1)
    states     = traj.states[1:end-1] |> to_matrix
    logprobs   = traj.logprobs        |> to_matrix |> logsoftmax
    actions    = traj.actions         |> to_matrix
    values_    = traj.values[1:end-1] |> to_matrix
    advantages = traj.advantages      |> to_matrix
    returns    = advantages .+ values_
    logprobs_old  = map(getindex, eachcol(logprobs), actions)

    ps = params(actor, critic)
    gs = gradient(ps) do
        # Clip loss
        logprobs_all = logsoftmax(actor(states))
        logprobs_theta = map(getindex, eachcol(logprobs_all), actions[:])
        ratio = exp.(logprobs_theta .- logprobs_old)
        clip1 = ratio .* advantages
        clip2 = clamp.(ratio, 1-ϵ, 1+ϵ)
        loss_clip = -mean(min.(clip1, clip2))

        # VF Loss
        values = critic(states)
        loss_vf = mean((returns .- values) .^ 2)
        loss_clip + loss_vf * vf_coeff
    end
    update!(opt, ps, gs)
end

function PPO(;num_workers=8, T=30)

    actor = Chain(
                Dense(4, 64, leakyrelu; init = Flux.glorot_uniform(rng)),
                Dense(64, 2; init = Flux.glorot_uniform(rng)))

    critic = Chain(
                Dense(4, 64, leakyrelu; init = Flux.glorot_uniform(rng)),
                Dense(64, 1; init = Flux.glorot_uniform(rng)))

    opt = ADAM()
    total_rewards = []
    workers = [CartPoleEnv() for _ in 1:num_workers]
    foreach(reset!, workers)

    function run_policy()
        traj = PPOTrajectory()
        # Instead of collecting a trajectory one worker
        # at a time, we vectorize the operations and
        # fetch info from all workers in one broadcast
        # so that each timestep has `num_workers` entries
        for t in 1:T
            push!(traj.states,    state.(workers))
            push!(traj.values,    critic.(traj.states[t]) .|> sum)
            push!(traj.logprobs,  actor.(traj.states[t]))
            push!(traj.actions,   sample.(softmax.(traj.logprobs[t])))
            push!(traj.rewards,   reward.(workers))
            push!(traj.dones,     done.(workers))
            # Step environment or reset it if done
            for n in 1:num_workers
                if traj.dones[end][n] == 1f0
                    reset!(workers[n])
                else
                    step!(workers[n], traj.actions[t][n])
                end
            end
            # step each worker with corresponding action
            foreach(step!, workers, traj.actions[t])
        end
        push!(traj.states,    state.(workers))
        push!(traj.values,    critic.(traj.states[T+1]) .|> sum)
        traj
    end

    for epoch in 1:100
        for i in 1:10
            traj = run_policy()
            compute_generalized_advantage_estimates!(traj)
            gs = update_weights!(opt, actor, critic, traj)
        end
        reward = test_models(actor, critic, N=5)
        push!(total_rewards, reward)
    end
    plot(total_rewards, title="Reward over Time", xlabel="Steps", ylabel="Reward",legend=false)


end

function test_models(actor, critic; N=10)
    total_reward = 0
    for _ in 1:N
        env = CartPoleEnv()
        state = reset!(env)
        avg_value = critic(state)[1]
        done = 0
        step = 0
        while done == 0f0
            probs = actor(state)
            action = sample(softmax(probs))
            state, reward, done, info = step!(env, action)
            step += 1
            total_reward += reward
            avg_value += critic(state)[1]
            step == 200 && break
        end
    end
    total_reward/N
end
