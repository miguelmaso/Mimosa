# We will take the best combination of nodes, experiments, layers and iterations (the one that gives the higher R2). We will import the weights and biases and create a ML model out of it. 
# Then, we will use one load combination [potential 1, potential 2, potentia 3, potential 4] and its results. We will feed the ML model with that input and we will compare its output againts the FE results.

# Recreate the ML model out of the weights and biases in the JSON file. We will take the nodes=10, experiments=4000, neurons=20 and layers=4 with an R2=0.941
using Flux
using JSON
using DelimitedFiles

#---------------------------------------------------------------------------
# Read the JSON file and create a model with the trained weights and biases
#---------------------------------------------------------------------------

model_json = open("Layers:4 Neurons:20 Experiments:4000 Nodes:10Iter:5000.json", "r") do file
    read(file, String)
end

# Parse the JSON data
model_data = JSON.parse(model_json)
architecture = model_data["architecture"]
weights = model_data["weights"]
bias = model_data["bias"]

# It is needed because JSON does not recognize Matrices, but only a vector of vectors of type Vector{Any}
function to_mat(arrs) # for lists-of-lists parsed by JSON
    return Matrix(hcat(Vector{Float32}.(arrs)...)')
end
# Function to create a model from architecture
    function create_model(architecture)
        layers = []
        for (i, layer_info) in enumerate(architecture)
            if layer_info[1] == "Dense"
                # Extract input and output dimensions from architecture tuple
                in_dim, out_dim = layer_info[2]

                # Special handling for first and last layers to correctly set dimensions
                if i == 1
                    layer = Dense(in_dim, out_dim,softmax)
                elseif i == length(architecture)
                    layer = Dense(in_dim, out_dim,softmax)
                else
                    layer = Dense(in_dim, out_dim,softmax)
                end

                push!(layers, layer)
            elseif layer_info == "ReLU"
                push!(layers, relu)
            elseif layer_info == "Softmax"
                push!(layers, softmax)
            end
        end
        return Chain(layers...)
    end

#model = create_model(architecture)
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
model = create_neural_network(4,20,4,20,softplus)
# Function to assign weights to model
function assign_weights!(model, weights)
    idx = 1
    for layer in model
        if isa(layer, Dense)
            n_weights = to_mat(weights[idx])' # Returns a Matrix{Float32}
            n_bias = bias[idx]
            layer.weight .= n_weights
            layer.bias .= convert(Vector{Float32},n_bias)
            idx += 1
        end
    end
end

assign_weights!(model, weights)
println("DONE creating the ML model")

#---------------------------------------
# Import a load and results combination
#---------------------------------------
x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
y_train::Matrix{Float64} = readdlm("contents_output.txt")

y_predicted = sort(model(x_train[1,:]))
y_fromFE    = sort(y_train[:,1])



