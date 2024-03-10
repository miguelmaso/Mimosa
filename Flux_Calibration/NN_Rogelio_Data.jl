using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train = readdlm("Input_Alberto_.txt", ',')
x_test = readdlm("Input_Alberto_.txt", ',')
y_train = readdlm("Output_Alberto_.txt", ',')
y_test = readdlm("Output_Alberto_.txt", ',')

x_train = x_train[1:30,:]'
y_train = y_train[1:30]'
x_test = x_test[1:30,:]'
y_test = y_test[1:30]'

#-------------------------------------------------------------------------------
# Build a model. Now it's just a simple layer with one input and one output
#-------------------------------------------------------------------------------
#Let's create a multi-layer perceptron
model = Chain(
    Dense(5=>3),
    BatchNorm(3),
    Dense(3=>3),
    BatchNorm(3),
    Dense(3 => 1)
)



#model = Dense(5 => 1) # Specified sigmoid activation function

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
opt = Descent(0.1) # Define an optimisation strategy. In this case, just the gradient descent. But could de Adams, etc. 
printstyled("The learning rate of the gradient descent is $opt \n"; color = :green)

data = [(x_train,y_train)]
# Now we iteratively train the model with the training data, minimizing the loss function by updating the weights and biases following the gradient descent
function iterative_training(model, x_train, y_train)
    epoch = 1
    iter = 1
    Losses = zeros(0)
    while  epoch<1000
     train!(loss, model, data, opt)
     L = loss(model, x_train, y_train)
     println("Epoch number $epoch with a loss $L ")
     iter += 1
     epoch += 1
     #-------------------
     #Store the gradients
     #-------------------
#     dLdm, dLdx, _= gradient(loss,model, x_test, y_test)
#     append!(Weights, dLdm.weight)
#     append!(Biases, dLdm.bias)
#     append!(GradientX, dLdx)
      append!(Losses,L)
    end
    return Losses
end

Losses=iterative_training(model, x_train, y_train)


#-------------------------------------------------------------------------------
# Predict the model
#-------------------------------------------------------------------------------
#_, dLdx, _= gradient(loss,model, x_test, y_test)
new_model(x) = sum(model(x)) # This is to obtain a scalar output for the model; which is a vector by default
grad = gradient(new_model, x_test)
#println(grad)

#printstyled("Predicted data is: \n"; color= :green)
#println(model(x_test))
#printstyled("Actual data is:\n"; color= :green)
#println(y_test)
plot([vec(y_train),vec(model(x_train))], seriestype = :scatter, label=["Original Test" "Fitting Results"])