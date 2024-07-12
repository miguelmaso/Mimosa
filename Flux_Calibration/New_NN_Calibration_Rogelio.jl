using Flux
using JSON
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
using LinearAlgebra
using Random
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
y_train::Matrix{Float64} = readdlm("contents_output.txt")
mat_coords::Matrix{Float64} = readdlm("mat_coords.txt")
mat_coords_shaped = reshape(mat_coords,(3,133))'

#x_train_n = x_train[:,1:2]

#-------------------------------------------------------------------------------
# Extract components 1, 2 and 3 of displacements
#-------------------------------------------------------------------------------
n_nodes   =  size(y_train,1)
y_train₁  =  y_train[1:3:n_nodes,:]
y_train₂  =  y_train[2:3:n_nodes,:]
y_train₃  =  y_train[3:3:n_nodes,:]

plot(y_train₁[1:2000],label="u₁")
plot!(y_train₂[1:2000],label="u₂")
plot!(y_train₃[1:2000],label="u₃")


@show size(mat_coords_shaped)


@inline function normalise(x::AbstractArray; dims=ndims(x), ϵ=1e-8)
#  μ = mean(x, dims=dims)
#  σ = std(x, dims=dims, mean=μ, corrected=false)
 #  return @. (x - μ) / (σ + ϵ)
 return x
end
#TODO Trabajar en un batch de datos para reducir el tiempo. 
# El batch lo puedes seleccionar con un MonteCarlo Sampling
#TODO Probar con una red más sencilla



x_train_whole= x_train'
y_train₁_whole= y_train₁
y_train₂_whole= y_train₂
y_train₃_whole= y_train₃


# Create batches of data to work with. 
# train_loader = Flux.Data.DataLoader((x_train_whole, y_train_whole), batchsize=20, shuffle=true)
# for (x_batch, y_batch) in train_loader
#     println(size(x_batch))  # Should print (4, batch_size)
#     println(size(y_batch))  # Should print (399, batch_s399ize)
#     break
# end

n_experiments            =  size(y_train₁_whole,2)
n_nodes                  =  size(y_train₁_whole,1)
n_experiments_training   =  min(200,n_experiments)
n_nodes_training         =  min(10,n_nodes)
training_indices         =  randperm(n_experiments)[1:n_experiments_training]
nodes_indices            =  randperm(n_nodes)[1:n_nodes_training]





function normalize(row::Vector)
    min = minimum(row)
    max = maximum(row)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,(row[i]-min)/(max-min))
    end
  return scaled
end

function normalize_columns(matrix::Matrix{Float64})::Matrix{Float64}
    rows, cols = size(matrix)
    normalized_matrix = Matrix{Float64}(undef, rows, cols)
    for j in 1:cols
        normalized_matrix[:, j] = normalize(matrix[:, j])
    end
    return normalized_matrix
end

x_train_norm        =  zeros(size(x_train_whole,1),size(x_train_whole,2))
for i in 1:4
  x_train_norm[i,:] = normalize(x_train_whole[i,:])
end
y_train₁_norm = reshape(normalize(y_train₁_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₃_norm = reshape(normalize(y_train₃_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train_norm = vcat(y_train₁_norm,y_train₃_norm)

n_components  =  2

x_train_batch   =  x_train_whole[:,training_indices]
y_train₁_batch   =  y_train₁_whole[nodes_indices,training_indices]
y_train₂_batch   =  y_train₂_whole[nodes_indices,training_indices]
y_train₃_batch   =  y_train₃_whole[nodes_indices,training_indices]
y_train_batch  = vcat(y_train₁_batch,y_train₃_batch)
# if all(>(0),y_train_norm) == false
#     error()
# end

#-------------------------------------------------------------------------------
# Build a model. Now it's just a simple layer with one input and one output
#-------------------------------------------------------------------------------
#Let's create a multi-layer perceptron
# ! We should have a function that recursively creates a model with n layers, m neurons and we can provide the activation function and initialization
#TODO INITIALIZE THE NN SO THAT THE FIRST OUTPUT IS 0 TO MAKE THE LOSS EQUAL TO 1
# model = Chain(
#    Dense(4=>200, softplus; bias=zeros(200), init=Flux.zeros32),
#    BatchNorm(200),
#    Dense(200=>200,softplus; bias=zeros(200), init=Flux.zeros32),
#    BatchNorm(200),
#    Dense(200=>200,softplus; bias=zeros(200), init=Flux.zeros32),
#    BatchNorm(200),
#    Dense(200=>200,softplus; bias=zeros(200), init=Flux.zeros32),
#    BatchNorm(200),
#    Dense(200=>200,softplus; bias=zeros(200), init=Flux.zeros32),
#    BatchNorm(200),
#    Dense(200=>399,softplus; bias=zeros(399), init=Flux.zeros32),
# )


function create_neural_network(input_size::Int, output_size::Int, hidden_layers::Int, neurons_per_layer::Int, activation::Function)
    # Create the layers
    layers = []

    # Input layer
    push!(layers, Dense(input_size, neurons_per_layer, activation))

    # Hidden layers
    for i in 2:hidden_layers
        push!(layers, Dense(neurons_per_layer, neurons_per_layer, activation))
    end

    # Output layer
    push!(layers, Dense(neurons_per_layer, output_size))

    # Create the model
    model = Chain(layers...)

    return model
end


model = create_neural_network(4,n_nodes_training*n_components,4,40,softplus)

#---------------------------------------------------------------------------------
# We need a way to store the structure and the trained model (weights and biases)
#---------------------------------------------------------------------------------
# Function to extract model architecture
function extract_architecture(model)
    architecture = []
    for layer in model
        if isa(layer, Dense)
            push!(architecture, ("Dense", size(layer.weight)))
        elseif isa(layer, typeof(relu))
            push!(architecture, "ReLU")
        elseif isa(layer, typeof(softmax))
            push!(architecture, "Softmax")
        end
    end
    return architecture
end

# Function to extract model weights
function extract_weights(params)
    weights = []
    for p in params
        push!(weights, p |> Array)
    end
    return weights
end
# Combine architecture and weights into a single JSON object
model_data = Dict("architecture" => architecture, "weights" => weights)

# Convert to JSON and save
model_json = JSON.json(model_data)

open("model.json", "w") do file
    write(file, model_json)
end

#-------------------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------------------

function loss(flux_model,x,y)
    ŷ = flux_model(x)

    num = sum((dot(ŷ-y,ŷ-y)))
    den = sum((dot(y,y)))

    return sqrt(num/den)
end
# function loss(flux_model,x,y)
#     ŷ = flux_model(x)
#     num = sum(dot(ŷ[:,i]-y[:,i],ŷ[:,i]-y[:,i])^2 for i in 1:size(y,2))
#     den = sum(dot(y[:,i],y[:,i])^2 for i in 1:size(y,2))
#     out = num/den
# end;
# function loss(flux_model,x,y)
#     ŷ = flux_model(x)
#     Flux.mse(ŷ,y)
# end;

#--------------------------------------------------------------------------------------------------------------
# We need to define what index in the results (predicted) values, are at the ends of the beam. They will be the
# most representatives, so we will use them to calculate the R2 error of the euclidean norm
#--------------------------------------------------------------------------------------------------------------

Y = [40.0,8.0,0.8] #Coordenates of point that is representative of the deformation.
#Euclidean norm between 2 vectors
function euclidean_norm(row,Y)
    return norm(row-Y)
end

# Compute distances for each row in X
distances = [euclidean_norm(mat_coords_shaped[i, :], Y) for i in 1:size(mat_coords_shaped, 1)]


# Find the index of the minimum distance
min_index = argmin(distances)


#Necesitamos definir el R2 en base a la norma euclidea entre 2 vectores

function r2d2(actual_values::Matrix, predicted_values::Matrix) # This function will take 2 matrices as input, as the variable to be trained is displacement

        Norms_actual = zeros(size(actual_values,2))
        Norms_predicted = zeros(size(actual_values,2))

    #Reshape and sanitize the input vectors
    for column in size(actual_values,2)
        actual_reshaped = reshape(actual_values[:,column],(3,133))'
        predicted_reshaped = reshape(predicted_values[:,column],(3,133))'
        Norm_actual = norm(actual_reshaped[min_index,:])
        Norm_predicted = norm(predicted_reshaped[min_index,:])
        append!(Norms_actual,Norm_actual)
        append!(Norms_predicted,Norm_predicted)

    end
    @show size(Norms_actual)
    # Compute the mean of actual values
    mean_actual = mean(Norms_actual)

    # Compute the sum of squares of residuals
    SS_res = sum((Norms_actual .- Norms_predicted).^2)

    # Compute the total sum of squares
    SS_tot = sum((Norms_actual .- mean_actual).^2)

    # Compute R-squared (R2) error
    R2 = 1 - SS_res / SS_tot

    return R2
end



function r2d2(actual_values::Vector, predicted_values::Vector) # This function will take 2 matrices as input, as the variable to be trained is displacement

    Norms_actual = zeros(size(actual_values,2))
    Norms_predicted = zeros(size(actual_values,2))

#Reshape and sanitize the input vectors
for column in size(actual_values,2)
    actual_reshaped = reshape(actual_values[:,column],(3,133))'
    predicted_reshaped = reshape(predicted_values[:,column],(3,133))'
    Norm_actual = norm(actual_reshaped[min_index,:])
    Norm_predicted = norm(predicted_reshaped[min_index,:])
    append!(Norms_actual,Norm_actual)
    append!(Norms_predicted,Norm_predicted)

end
@show size(Norms_actual)
# Compute the mean of actual values
mean_actual = mean(Norms_actual)

# Compute the sum of squares of residuals
SS_res = sum((Norms_actual .- Norms_predicted).^2)

# Compute the total sum of squares
SS_tot = sum((Norms_actual .- mean_actual).^2)

# Compute R-squared (R2) error
R2 = 1 - SS_res / SS_tot

return R2
end



function R2Function(actual_values, predicted_values) 
    dims           =  ndims(actual_values)
    if dims==1
       mean_actual    =  mean(actual_values, dims=dims)
       SS_res         =  sum((actual_values[:] - predicted_values[:]).^2)
       SS_tot         = sum(((actual_values[:] .- mean_actual).^2))
    else
        mean_actual    =  mean(actual_values, dims=dims)
        matrix         =  (actual_values .- predicted_values).^2
        SS_res         =  sum([norm(matrix[:, i]) for i in 1:size(matrix, 2)])
        matrix         =  (actual_values .- mean_actual).^2
        SS_tot         = sum([norm(matrix[:, i]) for i in 1:size(matrix, 2)])
     end
    R2             = 1   - SS_res / SS_tot
    return R2
end


initial_loss = loss(model, x_train_batch, y_train_batch)
printstyled("The initial loss is $initial_loss \n"; color = :red)
#opt = Descent(0.02) # Define an optimisation strategy. In this case, just the gradient descent. But could de Adams, etc. 
opt = Flux.setup(Adam(0.01), model)


# Now we iteratively train the model with the training data, minimizing the loss function by updating the weights and biases following the gradient descent
function iterative_training(model, x_train, y_train,maxIter)
    data = [(x_train,y_train)]
    epoch = 1
    iter = 1
    Losses = zeros(0)
    #while  loss(model,x_train,y_train)>0.00005
    while iter<maxIter
     train!(loss, model, data, opt)
     L = loss(model, x_train, y_train)
     println("Epoch number $epoch with a loss $L ")
     @show size(y_train)
     @show size(model(x_train))
     R2=R2Function(y_train[:],vec(model(x_train)))
     println("R2  is $R2")

     iter += 1
     epoch += 1
     #-------------------
     #Store the gradients
     #-------------------
#     dLdm, dLdx, _= gradient(loss,model, x_test, y_test)
#     append!(Weights, dLdm.weight)
#     append!(Biases, dLdm.bias)
#     append!(GradientX, dLdx)
#      plot([vec(y_train_norm),vec(model(x_train_norm))], seriestype = :scatter, label=["Original Test" "Fitting Results"])
      append!(Losses,L)
    end
    return model,Losses
end

maxIter   =  1e4

model, Losses=iterative_training(model, x_train_batch, y_train_batch, maxIter)



y_train₁_eval = y_train₁_norm[nodes_indices,:]
y_train₃_eval = y_train₃_norm[nodes_indices,:]
y_train_eval = vcat(y_train₁_eval,y_train₃_eval)
R2_new = R2Function(y_train_eval, model(x_train_norm))


plot(log.(Losses),label="log(Loss")
#-------------------------------------------------------------------------------
# Predict the model
#-------------------------------------------------------------------------------
#_, dLdx, _= gradient(loss,model, x_test, y_test)
#new_model(x) = sum(model(x)) # This is to obtain a scalar output for the model; which is a vector by default
#Piola = gradient(new_model, x_train)
#println(grad)


#-------------------------------------------------------------------------------
# Sort y_train and find its indices. Then use those indices in the predicted output
#-------------------------------------------------------------------------------

function sort_and_apply_indices(original_arr, apply_arr)
    # Create a copy of the original array
    sorted_arr = copy(original_arr)
    
    # Sort the array in ascending order
    sort!(sorted_arr,dims=2)
    
    # Find the indices that were changed
    indices_changed = sortperm(original_arr,dims=2)
    
    # Apply the indices to another array
    result_arr = similar(apply_arr, length(apply_arr))
    for (i, idx) in enumerate(indices_changed)
        result_arr[i] = apply_arr[idx]
    end
    
    return sorted_arr, indices_changed, result_arr
end

#y_predicted = model(x_train_norm)
#sorted_y_train, indices_y_train, sorted_y_predicted = sort_and_apply_indices(y_train_norm, y_predicted)






#printstyled("Predicted data is: \n"; color= :green)
#println(model(x_test))
#printstyled("Actual data is:\n"; color= :green)
#println(y_test)
#plot([vec(y_train_norm),vec(model(x_train_norm))], label=["Original Test" "Fitting Results"])

# num = ∑(y_i - y_i*)^2
# den = ∑(y_i*)^2
# y* es la analitica