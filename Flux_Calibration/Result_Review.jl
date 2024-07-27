using Flux
using JSON
using Base.Filesystem

# Function to list files in a directory
function list_files_in_directory(directory::String)
    files = []
    R2_max = 0
    Config = []
    for (root, dirs, file_names) in walkdir(directory)
        for file_name in file_names
#            push!(files, joinpath(file_name))
             model_json = open(joinpath(root,file_name),"r") do file
                read(file, String)
             end
             model_data = JSON.parse(model_json)
             R2 = model_data["R2"]
             if R2>R2_max
                R2_max = R2
                Config = file_name
             end

        end
    end
    return files, R2_max, Config
end

# Example usage
directory_path = "NN_parametric_run"
file_names, R2, File = list_files_in_directory(directory_path)

#---------------------------------------------------
# Load the model and double check the R2 value
#---------------------------------------------------
# Read the best JSON file
model_json = open(joinpath("NN_parametric_run",File), "r") do file
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
#model = create_neural_network(4,20,4,40,softplus)
model = create_model(architecture)
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
println("DONE")


#----------------------------------------------------------
# With the model trained, let's import the data and compare
#----------------------------------------------------------