using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train = readdlm("Input_Alberto_2.txt", ',')
x_test = readdlm("Input_Alberto_2.txt", ',')
y_train = readdlm("Output_Alberto_2.txt", ',')
y_test = readdlm("Output_Alberto_2.txt", ',')

x_train_n = hcat(x_train[:,1:2],x_train[:,4:5])
#x_train_n = x_train[:,1:2]
y_train_n= y_train[:,1]
@show size(x_train_n)
@show size(y_train_n)

x_train = x_train_n'
y_train = y_train_n'
x_test = x_test'
y_test = y_test'
@show size(x_train)
@show size(y_train)

@inline function normalise(x::AbstractArray; dims=ndims(x), ϵ=1e-8)
  μ = mean(x, dims=dims)
  σ = std(x, dims=dims, mean=μ, corrected=false)
 #  return @. (x - μ) / (σ + ϵ)
 return x
end

x_train_norm=normalise(x_train)
y_train_norm=normalise(y_train)




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
   Dense(6=>1,softplus),
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
    num = sum((ŷ - y).^2)
    den = sum((y.^2))
    out = num/den
end;
function r2d2(actual_values, predicted_values) 
    # Compute the mean of actual values
    mean_actual = mean(actual_values)

    # Compute the sum of squares of residuals
    SS_res = sum((actual_values .- predicted_values).^2)

    # Compute the total sum of squares
    SS_tot = sum((actual_values .- mean_actual).^2)

    # Compute R-squared (R2) error
    R2 = 1 - SS_res / SS_tot

    return R2
end





initial_loss = loss(model, x_train_norm, y_train_norm)
printstyled("The initial loss is $initial_loss \n"; color = :red)
#opt = Descent(0.02) # Define an optimisation strategy. In this case, just the gradient descent. But could de Adams, etc. 
opt = Flux.setup(Adam(0.008), model)
printstyled("The learning rate of the gradient descent is $opt \n"; color = :green)

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
     R2=r2d2(model(x_train_norm),y_train_norm)
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
      plot([vec(y_train_norm),vec(model(x_train_norm))], seriestype = :scatter, label=["Original Test" "Fitting Results"])
      append!(Losses,L)
    end
    return Losses
end

Losses=iterative_training(model, x_train_norm, y_train_norm)


#-------------------------------------------------------------------------------
# Predict the model
#-------------------------------------------------------------------------------
#_, dLdx, _= gradient(loss,model, x_test, y_test)
new_model(x) = sum(model(x)) # This is to obtain a scalar output for the model; which is a vector by default
Piola = gradient(new_model, x_train)
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

y_predicted = model(x_train_norm)
sorted_y_train, indices_y_train, sorted_y_predicted = sort_and_apply_indices(y_train_norm, y_predicted)






#printstyled("Predicted data is: \n"; color= :green)
#println(model(x_test))
#printstyled("Actual data is:\n"; color= :green)
#println(y_test)
plot([vec(y_train_norm),vec(model(x_train_norm))], label=["Original Test" "Fitting Results"])

# num = ∑(y_i - y_i*)^2
# den = ∑(y_i*)^2
# y* es la analitica