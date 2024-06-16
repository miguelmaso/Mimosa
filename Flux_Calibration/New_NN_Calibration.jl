using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
using LinearAlgebra
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
y_train::Matrix{Float64} = readdlm("contents_output.txt")
mat_coords::Matrix{Float64} = readdlm("mat_coords.txt")
mat_coords_shaped = reshape(mat_coords,(3,133))'

#x_train_n = x_train[:,1:2]
@show size(x_train)
@show size(y_train)
@show size(mat_coords_shaped)


@inline function normalise(x::AbstractArray; dims=ndims(x), ϵ=1e-8)
#  μ = mean(x, dims=dims)
#  σ = std(x, dims=dims, mean=μ, corrected=false)
 #  return @. (x - μ) / (σ + ϵ)
 return x
end

x_train_norm= x_train'
y_train_norm= y_train




#-------------------------------------------------------------------------------
# Build a model. Now it's just a simple layer with one input and one output
#-------------------------------------------------------------------------------
#Let's create a multi-layer perceptron
model = Chain(
   Dense(4=>6, softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>399,softplus),
)



#model = Dense(5 => 1)  

#-------------------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------------------
# function loss(flux_model, x, y) # Alternative definition using Flux's version of the MSE
#     ŷ = flux_model(x)
#     Flux.mse(ŷ, y)
# end;
function loss(flux_model,x,y)
    ŷ = flux_model(x)
    num = sum((dot(ŷ,y)).^2)
    den = sum((dot(y,y)))
    out = num/den
end;

#--------------------------------------------------------------------------------------------------------------
# We need to define what index in the results (predicted) values, are at the ends of the beam. They will be the
# most representatives, so we will use them to calculate the R2 error of the euclidean norm
#--------------------------------------------------------------------------------------------------------------
Y = [1.0,2.0,3.0]
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
        Norm_actual = actual_reshaped[min_index,:]
        Norm_predicted = predicted_reshaped[min_index,:]
        append!(Norms_actual,Norm_actual)
        append!(Norms_predicted,Norm_predicted)

    end

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


using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
using LinearAlgebra
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
y_train::Matrix{Float64} = readdlm("contents_output.txt")
mat_coords::Matrix{Float64} = readdlm("mat_coords.txt")
mat_coords_shaped = reshape(mat_coords,(3,133))'

#x_train_n = x_train[:,1:2]
@show size(x_train)
@show size(y_train)
@show size(mat_coords_shaped)


@inline function normalise(x::AbstractArray; dims=ndims(x), ϵ=1e-8)
#  μ = mean(x, dims=dims)
#  σ = std(x, dims=dims, mean=μ, corrected=false)
 #  return @. (x - μ) / (σ + ϵ)
 return x
end

x_train_norm= x_train'
y_train_norm= y_train




#-------------------------------------------------------------------------------
# Build a model. Now it's just a simple layer with one input and one output
#-------------------------------------------------------------------------------
#Let's create a multi-layer perceptron
model = Chain(
   Dense(4=>6, softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>6,softplus),
   BatchNorm(6),
   Dense(6=>399,softplus),
)



#model = Dense(5 => 1)  

#-------------------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------------------
# function loss(flux_model, x, y) # Alternative definition using Flux's version of the MSE
#     ŷ = flux_model(x)
#     Flux.mse(ŷ, y)
# end;
function loss(flux_model,x,y)
    ŷ = flux_model(x)
    num = sum((dot(ŷ,y)).^2)
    den = sum((dot(y,y)))
    out = num/den
end;

#--------------------------------------------------------------------------------------------------------------
# We need to define what index in the results (predicted) values, are at the ends of the beam. They will be the
# most representatives, so we will use them to calculate the R2 error of the euclidean norm
#--------------------------------------------------------------------------------------------------------------
Y = [1.0,2.0,3.0]
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


initial_loss = loss(model, x_train_norm, y_train_norm)
printstyled("The initial loss is $initial_loss \n"; color = :red)
#opt = Descent(0.02) # Define an optimisation strategy. In this case, just the gradient descent. But could de Adams, etc. 
opt = Flux.setup(Adam(0.05), model)

data = [(x_train_norm,y_train_norm)]
# Now we iteratively train the model with the training data, minimizing the loss function by updating the weights and biases following the gradient descent
function iterative_training(model, x_train, y_train)
    epoch = 1
    iter = 1
    Losses = zeros(0)
    while  loss(model,x_train,y_train)>0.00005
     train!(loss, model, data, opt)
     L = loss(model, x_train, y_train)
     println("Epoch number $epoch with a loss $L ")
     R2=r2d2(y_train_norm,model(x_train_norm))
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
    return Losses
end

Losses=iterative_training(model, x_train_norm, y_train_norm)


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