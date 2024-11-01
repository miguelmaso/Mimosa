# We will take one node of the FE and paint its trajectory through the loadsteps. Then, we will compare that against the ML prediction for each loadstep

# Let's take experiment 18673 as a reference (it was the one that we plotted the R2 on)

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

theme(:wong2,fontfamily="Courier")

cd("/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/NN_parametric_run")
#---------------------------------------------------------------------------
# Read the JSON file and create a model with the trained weights and biases
#---------------------------------------------------------------------------
model_json = open("Layers:4 Neurons:20 Experiments:2000 Nodes:10Iter:5000 Corrected.json", "r") do file
    read(file, String)
end

# Parse the JSON data
model_data = JSON.parse(model_json)
architecture = model_data["architecture"]
weights = model_data["weights"]
bias = model_data["bias"]
nodes_indices = model_data["Node_Indices"]
experiment_indices = model_data["Training_Indices"]
Losses = model_data["Losses"]

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
cd("/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/ParsingScripts")
#---------------------------------------
# Import a load and results combination
#---------------------------------------
x_train_subset::Matrix{Float64} = readdlm("filenames_parsed_FE_Trajectory_New.txt")
#x_train_subset::Matrix{Float64} = readdlm("filenames_parsed_FE_Trajectory.txt")
x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
y_train_subset::Matrix{Float64} = readdlm("contents_output_FE_Trajectory_New.txt")
#y_train_subset::Matrix{Float64} = readdlm("contents_output_FE_Trajectory.txt")
y_train::Matrix{Float64} = readdlm("contents_output.txt")
n_nodes   =  size(y_train,1)
y_train₁  =  y_train[1:3:n_nodes,:]
y_train₂  =  y_train[2:3:n_nodes,:]
y_train₃  =  y_train[3:3:n_nodes,:]


# I will use the whole training available for the normalization. However, I will select only a subset of those, which correspond to a potential and its loadsteps
# for the trajectory plotting. To select a subset of the available training data, I will retrieve the indices of the whole training that match to those of the potential and its loadsteps that I chose
function find_matching_rows(small_matrix, big_matrix)
    # Ensure both matrices have the same number of columns
    if size(small_matrix, 2) != size(big_matrix, 2)
        error("Both matrices must have the same number of columns.")
    end
    
    # Initialize an empty array to store matching row indices
    matching_indices = []

    # Iterate over rows of the big matrix
    for i in 1:size(big_matrix, 1)
        # Check if the current row of the big matrix matches any row in the small matrix
        for j in 1:size(small_matrix, 1)
            if big_matrix[i, :] == small_matrix[j, :]
                push!(matching_indices, i)  # Save the index of the matching row
            end
        end
    end

    return matching_indices
end


comparing_index = find_matching_rows(x_train_subset,x_train)


x_train_whole= x_train'
y_train₁_whole= y_train₁
y_train₂_whole= y_train₂
y_train₃_whole= y_train₃

function normalise(row::Vector)
    min = minimum(row)
    max = maximum(row)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,(row[i]-min)/(max-min))
    end
  return scaled
end
# Recheck, only for my own clarification, how the training is defined (im not clear about the epochs and iterations)
x_train_norm = x_train_whole
y_train₁_norm_old = reshape(normalise(y_train₁_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₁_norm = map(x -> Float64(x), y_train₁_norm_old)
y_train₃_norm_old = reshape(normalise(y_train₃_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₃_norm = map(x -> Float64(x), y_train₃_norm_old)


y_train₁_eval = y_train₁_norm[nodes_indices,comparing_index]
y_train₃_eval = y_train₃_norm[nodes_indices,comparing_index]
y_train_eval = vcat(y_train₁_eval,y_train₃_eval)
x_train_subset = x_train[comparing_index,:]


# Let's take one point to plot its trajectory. First, we need to order the data in asceding order on the potential (x_train) and apply that to the results. It is not like that by default
# The distance that we plot should be either the norm, or Coord1 and Coord3 separately.
y_predicted = model(x_train_subset')
y_fromFE    = y_train_eval

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

R2_test = R2Function(vec(y_fromFE),vec(y_predicted))# Just to check that we are importing and treating the data properly. Since the point is from the training, the R2 should be high
#---------------------
# Trajectory plotting
#---------------------

# Sort x_train in ascending order and apply that to the displacements from FE and predicted
sorted_indices = sortperm(eachrow(x_train_subset))

y_fromFE_sorted = y_fromFE[:,sorted_indices]
y_predicted_sorted = y_predicted[:,sorted_indices]

#y_fromFE_sorted = y_fromFE
#y_predicted_sorted = y_predicted

# We need to take one point; ie: the 5 point. And we need to take Coord1 and Coord3 of that point 
Coord1_y_from_FE_sorted_point = y_fromFE_sorted[1:10,:]
Coord3_y_from_FE_sorted_point = y_fromFE_sorted[11:20,:]
Coord1_y_predicted_sorted_point = y_predicted_sorted[1:10,:]
Coord3_y_predicted_sorted_point = y_predicted_sorted[11:20,:]

# Plot per coordinate
plot(Coord3_y_from_FE_sorted_point[1:end],seriestype=:scatter,markersize=2)
plot!(Coord3_y_predicted_sorted_point[1:end],seriestype=:scatter,markersize=2)
# Plot everything together
plot(y_fromFE_sorted[1:end],seriestype=:scatter,markersize=2)
plot!(y_predicted_sorted[1:end],seriestype=:scatter,markersize=2)

# Data to Paraview CSV plotting
# We need to de-scale the data back

function denormalise(row::Vector,original_array)
    min = minimum(original_array)
    max = maximum(original_array)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,min + (max-min)*row[i])
    end
  return scaled
end

# We are selecting only 1 point, the first one
Coord1_y_predicted_sorted_point_descaled = denormalise(Coord1_y_predicted_sorted_point[1,:],y_train₁_whole[:])
Coord3_y_predicted_sorted_point_descaled = denormalise(Coord3_y_predicted_sorted_point[1,:],y_train₃_whole[:])

# We have to add the material coordinates to the result values that we get from the ML
# It is stored in the mat_coords.txt file, which holds the material coordinates for every point in the surface

# TODO The scaling works great. There is a mistake in the FE generation. The maximum in Paraview is not the same as the maximum that is stored in the contents_output_FE_Trajectory.txt
# TODO plus, there number of files generated during the FE generation (480) is NOT the same as the number of VTUs generated (531)
mat_coords = readdlm("mat_coords.txt")
mat_coords_reshape = reshape(mat_coords,3,133)
Point_41_MatCoords = mat_coords_reshape[:,41]
writedlm("PlottingTrajectoryParaview.csv", hcat(Coord1_y_predicted_sorted_point_descaled.+Point_41_MatCoords[1],zeros(480).+Point_41_MatCoords[2],Coord3_y_predicted_sorted_point_descaled.+Point_41_MatCoords[3]),",")