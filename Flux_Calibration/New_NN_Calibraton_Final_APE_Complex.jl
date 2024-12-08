using Flux
using JSON
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
using LinearAlgebra
using Random
using IterTools
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
# Remove any non numeric data
function remove_data(matrix1,matrix2)
    failed_rows = []
    for i in 1:size(matrix1,1)
        row = matrix1[i,:]
        if row[1] isa SubString
            append!(failed_rows,i)
        end
    end
    rows_to_keep = setdiff(1:size(matrix1,1),failed_rows)
    columns_to_keep = setdiff(1:size(matrix2,2),failed_rows)
    new_matrix1 = matrix1[rows_to_keep,:]
    new_matrix2 = matrix2[:,columns_to_keep]
    return failed_rows, new_matrix1, new_matrix2
end
cd("ParsingScripts/")
input_x_train = readdlm("filenames_parsed_complex_potential.txt")
input_y_train = readdlm("contents_output_complex_potential.txt")
cd("..")
failed_rows, x_train::Matrix{Float64}, y_train::Matrix{Float64} = remove_data(input_x_train,input_y_train')

#x_train::Matrix{Float64} = readdlm("filenames_parsed_corrected_Rogelio.txt")
#y_train::Matrix{Float64} = readdlm("contents_output_corrected_Rogelio.txt")
mat_coords::Matrix{Float64} = readdlm("mat_coords.txt")
mat_coords_shaped = reshape(mat_coords,(3,133))'

#x_train_n = x_train[:,1:2]

#-------------------------------------------------------------------------------
# Extract components 1, 2 and 3 of displacements
#-------------------------------------------------------------------------------
nodes   =  size(y_train,1)
y_train₁  =  y_train[1:3:nodes,:]
y_train₂  =  y_train[2:3:nodes,:]
y_train₃  =  y_train[3:3:nodes,:]

plot(y_train₁[1:2000],label="u₁")
plot!(y_train₂[1:2000],label="u₂")
plot!(y_train₃[1:2000],label="u₃")









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


function parametric_run(n_layers,n_neurons,n_experiments,n_nodes,epochs)

name                     =  string("Layers:",n_layers," ", "Neurons:",n_neurons," ","Experiments:",n_experiments," ","Nodes:",n_nodes,"","Iter:",epochs," ","Corrected"  )
#n_experiments            =  size(y_train₁_whole,2)
#n_nodes                  =  size(y_train₁_whole,1)
#n_experiments_training   =  min(200,n_experiments)
n_experiments_training   =  n_experiments
#n_experiments_training   =  size(y_train₁_whole,2)
#n_nodes_training         =  min(10,n_nodes)
n_nodes_training         =  n_nodes
#n_nodes_training         =  size(y_train₁_whole,1)
training_indices         =  randperm(size(y_train₁_whole,2))[1:n_experiments]
nodes_indices            =  randperm(size(y_train₁_whole,1))[1:n_nodes]



function normalize_columns(matrix::Matrix{Float64})::Matrix{Float64}
    rows, cols = size(matrix)
    normalized_matrix = Matrix{Float64}(undef, rows, cols)
    for j in 1:cols
        normalized_matrix[:, j] = normalize(matrix[:, j])
    end
    return normalized_matrix
end

function normalize(row::Vector)
    min = minimum(row)
    max = maximum(row)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,(row[i]-min)/(max-min))
    end
  return scaled
end
# ----------------------------------------------
# Normalizing the input data (evaluation set)
# ----------------------------------------------
x_train_norm = x_train_whole
y_train₁_norm_old = reshape(normalize(y_train₁_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₁_norm = map(x -> Float64(x), y_train₁_norm_old)
y_train₃_norm_old = reshape(normalize(y_train₃_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₃_norm = map(x -> Float64(x), y_train₃_norm_old)
#y_train₁_norm = reshape((y_train₁_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
#y_train₃_norm = reshape((y_train₃_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train_norm = vcat(y_train₁_norm,y_train₃_norm)

#TODO Investigar por que va más lento normalizando. Tienes que normalizar las y. Si normalizas el y_train, el y_eval lo tienes que normalizar tambien.
n_components  =  2

x_train_batch   =  x_train_whole[:,training_indices]
y_train₁_batch   =  y_train₁_norm[nodes_indices,training_indices]
y_train₃_batch   =  y_train₃_norm[nodes_indices,training_indices]
#y_train₁_batch   =  y_train₁_whole[nodes_indices,training_indices]
#y_train₃_batch   =  y_train₃_whole[nodes_indices,training_indices]

y_train_batch  = vcat(y_train₁_batch,y_train₃_batch)
# if all(>(0),y_train_norm) == false
#     error()
# end

#------------------
# Build a model. 
#------------------



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


model = create_neural_network(20,n_nodes_training*n_components,n_layers,n_neurons,softplus)

#---------------------------------------------------------------------------------
# We need a way to store the structure and the trained model (weights and biases)
#---------------------------------------------------------------------------------
# Function to extract model architecture


function extract_architecture(model)
    architecture = []
    for (i, layer) in enumerate(model)
        if isa(layer, Dense)
            if i == 1
                push!(architecture, ("Dense", (size(layer.weight, 2), size(layer.weight, 1))))
            elseif i == length(model)
                push!(architecture, ("Dense", (size(layer.weight, 2), size(layer.weight, 1))))
            else
                push!(architecture, ("Dense", (size(layer.weight, 1), size(layer.weight, 2))))
            end
        elseif isa(layer, typeof(relu))
            push!(architecture, "ReLU")
        elseif isa(layer, typeof(softmax))
            push!(architecture, "Softmax")
        end
    end
    return architecture
end

# Function to extract model weights
function extract_weights(model)
    weights = []
    bias = []
    for layer in model
        if isa(layer, Dense)
            push!(weights, layer.weight)
            push!(bias, layer.bias)
        end
    end
    return weights, bias
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
     #@show size(y_train)
     #@show size(model(x_train))
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

maxIter   =  epochs

model, Losses=iterative_training(model, x_train_batch, y_train_batch, maxIter)



y_train₁_eval = y_train₁_norm[nodes_indices,:]
y_train₃_eval = y_train₃_norm[nodes_indices,:]
y_train_eval = vcat(y_train₁_eval,y_train₃_eval)
R2_new = R2Function(vec(y_train_eval), vec(model(x_train_norm)))
println("R2 new is:$R2_new")


#plot(log.(Losses),label="log(Loss")

#----------------------------------------------------------------------------------------------------------
# Store the weights and architecture of the trained model. Store the Loss and the R2 as well in a JSON file
#----------------------------------------------------------------------------------------------------------

architecture = extract_architecture(model)
weights,bias = extract_weights(model)


# Combine architecture and weights into a single JSON object
model_data = Dict("architecture" => architecture, "weights" => weights, "bias" => bias, "Losses"=> Losses, "R2"=>R2_new, "Node_Indices"=>nodes_indices, "Training_Indices"=>training_indices)


# Convert to JSON and save
model_json = JSON.json(model_data)

open(name*".json", "w") do file
    write(file, model_json)
end

return #vec(y_train_eval), vec(model(x_train_norm)), x_train_norm, x_train_batch,y_train_batch, model

end

# ---------------------------------------------
# Create a run of all the possible combinations
# ---------------------------------------------
# Define the parameter values
n_layers_values = [4,8,10]
n_neurons_values = [10,20,40]
n_experiments_values = [2077,7000,10000]
n_nodes_values = [50,100,200]
epochs_values = [1e4]
# TODO Elegir los nodos (los indices),al principio antes de lanzar el experimento
# TODO Forma de presentar los datos: para 4 combinaciones de datos (n de nodos y de experimentos), una tabla con el R2 en función de cada combinacion de layers y neuronas 
# TODO Graficas de correlación: coger el mejor R2 (la mejor arquitectura) de las 4 tablas de las 4 configuraciones de datos (El TODO 2). Plotear los desplazamientos de la predicción y del test (debería salir una recta si el R2 es alto). 
# TODO Plotear el logaritmo decimal de la loss para varios casos
# TODO Coger 4 casos (4 configuraciones de potencial), y lanzo un FEM de ese potencial. De ese FEM, guardo cada load increment. Entonces me fijo en uno de los nodos de la cara donde estoy proyectando en la superficie, y comparo el desplazamiento del nodo FEM con el desplazamietno del nodo correspondiente que te daría la red. Estas serían las gráfica de trayectorias
# TODO Pintar en Paraview el FEM y compararlo con la nube de puntos que te da la predicción de ML
# TODO Preparar un modelo más complejo (con 10 y 10 electrodos en las caras, u 8, pero que sea divisible por el numero de nodos para que no se quede ninguno en medio; que sea 4x2 o 5x2 en cada cara), adaptar el DOE para que lances eso 
# Generate all possible combinations of parameter values
combinations = IterTools.product(n_layers_values, n_neurons_values, n_experiments_values, n_nodes_values, epochs_values)


cd("NN_parametric_run_complex")

# Iterate through the combinations and call the parametric_run function
for combo in combinations
    n_layers, n_neurons, n_experiments, n_nodes, epochs = combo
    parametric_run(n_layers, n_neurons, n_experiments, n_nodes, epochs)
end

#y_eval,y_predicted, x_eval, x_batch, y_batch, model = parametric_run(4,40,2000,100)