using Printf
mutable struct Opt_State
    iteration::Int
    particles::Int
    n::Int
    X::Vector{Vector{Bool}}
    V_c::Vector{Vector{Float64}}
    V_0::Vector{Vector{Float64}}
    V_1::Vector{Vector{Float64}}
    P_ibest::Vector{Vector{Bool}}
    P_gbest::Vector{Bool}
    F_P_ibest::Vector{Float64}
    F_P_gbest::Float64
end

function InitialState(
    particles,
    Initial_X,
    Objective
    )
    iteration = 0
    n = length(Initial_X)
    F_P_gbest = Objective(Initial_X)
    X = [Int.(round.(rand(n),digits=0)) for _ in 1:particles]
    V_c = [[0.0 for _ in 1:n] for _ in 1:particles]
    V_0 = [[0.0 for _ in 1:n] for _ in 1:particles]
    V_1 = [[0.0 for _ in 1:n] for _ in 1:particles]
    P_ibest = [Vector{Bool}(undef,n) for _ in 1:particles]
    P_gbest = Initial_X
    F_P_ibest = Vector{Float64}(undef,particles)
    state = Opt_State(
        iteration,
        particles,
        n,
        X,
        V_c,
        V_0,
        V_1,
        P_ibest,
        P_gbest,
        F_P_ibest,
        F_P_gbest
    )
    return state
end

function compute_objective(X,Objective)
    n = length(X)
    F_X = Vector{Float64}(undef,n)
    Threads.@threads for i in 1:n
        F_X[i] = Objective(X[i])
    end
    return F_X
end

sig(x) = 1/(1 + exp(-x))

function update_state!(state,Objective,w,c1,c2)
    F_X = compute_objective(state.X,Objective)
    r1 = rand()
    r2 = rand()
    if state.iteration == 0
        state.F_P_ibest = copy(F_X)
        state.P_ibest = copy(state.X)
        F_X_min = minimum(F_X)
        X_min = state.X[argmin(F_X)]
        if F_X_min<state.F_P_gbest
            state.F_P_gbest = copy(F_X_min)
            state.P_gbest = copy(X_min)
        end
        # display(rand(state.particles))
        # display(state.V_0)
        # copyto!([rand(state.particles) for _ in 1:state.particles],state.V_0)
        # copyto!([rand(state.particles) for _ in 1:state.particles],state.V_1)
        # state.V_0 = [[10.0 for _ in 1:state.n] for _ in 1:state.particles]
        # state.V_1 = [[10.0 for _ in 1:state.n] for _ in 1:state.particles]
    else
        for i in 1:state.particles
            if F_X[i]<state.F_P_ibest[i]
                state.F_P_ibest[i] = copy(F_X[i])
                state.P_ibest[i] = copy(state.X[i])
            end
            if F_X[i]<state.F_P_gbest
                state.F_P_gbest = copy(F_X[i])
                state.P_gbest = copy(state.X[i])
                
            end
            for j in 1:state.n
                if state.P_ibest[i][j]
                    d11 = c1*r1
                    d01 = -c1*r1
                else
                    d01 = c1*r1
                    d11 = -c1*r1
                end
                if state.P_gbest[j]
                    d12 = c2*r2
                    d02 = -c2*r2
                else
                    d02 = c2*r2
                    d12 = -c2*r2
                end
                state.V_1[i][j] = w*state.V_1[i][j] + d11 + d12
                state.V_0[i][j] = w*state.V_0[i][j] + d01 + d02
                if state.X[i][j]
                    state.V_c[i][j] = state.V_0[i][j]
                else
                    state.V_c[i][j] = state.V_1[i][j]
                end
                if rand()<sig(state.V_c[i][j])
                    if state.X[i][j]
                        state.X[i][j] = false
                    else
                        state.X[i][j] = true
                    end
                end
            end
        end
    end
    state.iteration +=1
    return state
end

function optimize_BPSO(particles,
    Initial_X,
    Objective,w,c1,c2,iterations)

    state = InitialState(
    particles,
    Initial_X,
    Objective
    )
    # println("Iteration = $(state.iteration) --- Objective = $(state.F_P_gbest) --- Position = $(state.P_gbest) --- $(round.(sig.(sum(state.V_c)/state.particles),digits=2)) --- $(round(sum(round.(sig.(sum(state.V_c)/state.particles),digits=2))/state.n,digits=2))")
    while state.iteration<iterations
        state = update_state!(state,Objective,w,c1,c2)
        avg_location = round(sum(bin_to_int.(state.X))/state.particles,digits=0)
        max_location = maximum(bin_to_int.(state.X))
        minimum_location = minimum(bin_to_int.(state.X))
        range_location = "$max_location / $minimum_location"
        # print("\nIteration = $(state.iteration) --- Objective = $(state.F_P_gbest) --- Position = $(state.P_gbest) --- $(round.(sig.(sum(state.V_c)/state.particles),digits=2)) --- $(round(sum(round.(sig.(sum(state.V_c)/state.particles),digits=2))/state.n,digits=2))")
        
        # print("\nIteration = $(state.iteration) --- Objective = $(state.F_P_gbest) --- Position = $(state.P_gbest) --- $(round.((sum(state.V_c)/state.particles),digits=2)) --- $(round(sum(round.((sum(state.V_c)/state.particles),digits=2))/state.n,digits=2))")
        print("\nIteration = $(state.iteration) --- F_P_gbest = $(@sprintf "%0.2E" state.F_P_gbest) ---  P_gbest = $(bin_to_int(state.P_gbest)) --- Avg loc = $avg_location --- range loc = $range_location")
    end
    print("\n")
    return state
end


# functions required to test the algorithm

function bin_to_int16(bin)
    num = 0
    for i in 1:16
        if bin == digits(i-1,base=2,pad=4)
            num = i
        end
    end
    return num
end

function bin_to_int(bin)
    n = length(bin)
    num = 0
    count = 1
    for i in bin
        num += i*2^(n-count)
        count += 1
    end
    return num
end

Objective(x) = (bin_to_int(x) - 100)^2