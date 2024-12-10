using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
actual(x) = 4x +2
x_train, x_test = hcat(0:5...), hcat(6:10...)
#x_train, x_test = [0.0,1.0,2.0,3.0,4.0,5.0], [6.0,7.0,8.0,9.0,10.0]
y_train, y_test = actual.(x_train), actual.(x_test)

#-------------------------------------------------------------------------------
# Build a model. Now it's just a simple layer with one input and one output
#-------------------------------------------------------------------------------
model = Dense(1 => 1) # Specified sigmoid activation function
model.weight # Initialized weight
model.bias   # Initialized bias


#-------------------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------------------
#loss(model, x, y) = mean(abs2.(model(x) .- y)); #Loss function. This is the MSE, but could be a Flux.Crossentropy or any other function
function loss(flux_model, x, y) # Alternative definition using Flux's version of the MSE
    ŷ = flux_model(x)
    Flux.mse(ŷ, y)
end;
function evaluation(flux_model, x ) # Alternative definition using Flux's version of the MSE
    y_hat = flux_model(x)
    return y_hat
end;
initial_loss = loss(model, x_train, y_train)
printstyled("The initial loss is $initial_loss \n"; color = :red)
opt = Descent() # Define an optimisation strategy. In this case, just the gradient descent. But could de Adams, etc.
printstyled("The learning rate of the gradient descent is $opt \n"; color = :green)

data = [(x_train,y_train)]
# Now we iteratively train the model with the training data, minimizing the loss function by updating the weights and biases following the gradient descent
function iterative_training(model, x_train, y_train, tol)
epoch = 1
iter = 1
Weights = zeros(0)
Biases = zeros(0)
GradientX = zeros(0)
    while loss(model, x_train, y_train) > tol && iter<500
     train!(loss, model, data, opt)
     L = loss(model, x_train, y_train)
#     println("Epoch number $epoch with a loss $L ")
     iter += 1
     epoch += 1
     #-------------------
     #Store the gradients
     #-------------------
     dLdm, dLdx, _= gradient(loss,model, x_test, y_test)
     append!(Weights, dLdm.weight)
     append!(Biases, dLdm.bias)
     append!(GradientX, dLdx)
    end
return Weights, Biases, GradientX
end
Weights , Biases, GradientX= iterative_training(model, x_train, y_train, 1e-6)

#-------------------------------------------------------------------------------
# Predict the model
#-------------------------------------------------------------------------------
#_, dLdx, _= gradient(loss,model, x_test, y_test)
new_model(x) = sum(model(x)) # This is to obtain a scalar output for the model; which is a vector by default
grad = gradient(new_model, x_test)
println(grad)

#printstyled("Predicted data is: \n"; color= :green)
#println(model(x_test))
#printstyled("Actual data is:\n"; color= :green)
#println(y_test)
#plot(vec(x_test),[vec(y_test),vec(model(x_test))], seriestype = :scatter, label=["Original Test" "Fitting Results"])
