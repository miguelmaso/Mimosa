# We will take the best combination of nodes, experiments, layers and iterations (the one that gives the higher R2). We will import the weights and biases and create a ML model out of it. 
# Then, we will use one load combination [potential 1, potential 2, potentia 3, potential 4] and its results. We will feed the ML model with that input and we will compare its output againts the FE results.

# Recreate the ML model out of the weights and biases in the JSON file. We will take the nodes=10, experiments=4000, neurons=20 and layers=4 with an R2=0.941
using Flux
using JSON
using DelimitedFiles
using Plots
using Statistics
theme(:wong2,fontfamily="Courier")

cd("/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/NN_parametric_run_corrected_V3_copy/")
#---------------------------------------------------------------------------
# Read the JSON file and create a model with the trained weights and biases
#---------------------------------------------------------------------------
model_json = open("Layers:8 Neurons:40 Experiments:10000 Nodes:200Iter:10000 Corrected.json", "r") do file
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
model = create_neural_network(4,400,8,40,softplus)
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

input_x_train = readdlm("filenames_parsed_corrected_Rogelio.txt")
input_y_train = readdlm("contents_output_corrected_Rogelio.txt")

failed_rows, x_train::Matrix{Float64}, y_train::Matrix{Float64} = remove_data(input_x_train,input_y_train)
x_train_subset::Matrix{Float64} = readdlm("filenames_parsed_FE_Trajectory_New.txt")

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
#x_train::Matrix{Float64} = readdlm("filenames_parsed.txt")
#y_train::Matrix{Float64} = readdlm("contents_output.txt")

n_nodes   =  size(y_train,1)
y_train₁  =  y_train[1:3:n_nodes,:]
y_train₂  =  y_train[2:3:n_nodes,:]
y_train₃  =  y_train[3:3:n_nodes,:]

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

x_train_norm = x_train_whole
y_train₁_norm_old = reshape(normalise(y_train₁_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₁_norm = map(x -> Float64(x), y_train₁_norm_old)
y_train₃_norm_old = reshape(normalise(y_train₃_whole[:]),size(y_train₁_whole,1),size(y_train₁_whole,2))
y_train₃_norm = map(x -> Float64(x), y_train₃_norm_old)


y_train₁_eval = y_train₁_norm[nodes_indices,comparing_index]
y_train₃_eval = y_train₃_norm[nodes_indices,comparing_index]
y_train_eval = vcat(y_train₁_eval,y_train₃_eval)
x_train_subset = x_train[comparing_index,:]

#Test_point = 18673
# Test_point = 15
# validate = Test_point ∈ experiment_indices
# if validate == true
#     error("The test point belongs to the training")
# end
#y_predicted = model(x_train[Test_point,:])
y_predicted = model(x_train_subset')
#y_fromFE    = y_train_eval[:,Test_point]
y_fromFE    = y_train_eval

# --------------------------------
# Let's plot the R2
# --------------------------------

random_indices = rand(1:20577,2000)
y_pred_whole = model(x_train')
y_fromFE₁_whole = y_train₁_norm[nodes_indices,random_indices]
y_fromFE₃_whole = y_train₃_norm[nodes_indices,random_indices]
y_pred₁ = y_pred_whole[1:200,random_indices]
y_pred₃ = y_pred_whole[201:400,random_indices]
plot(y_pred₁[:],y_fromFE₁_whole[:],seriestype=:scatter, markersize=0.5, markershape=:circle,label="Displacement in Coord 1 ",legendfontsize=7,tickfontsize=9,guidefontsize=9,xlabel="Displacement from ML prediction",ylabel="Displacement from FE")
savefig("R2_Coord1_corrected_V3.pdf")
plot(y_pred₃[:],y_fromFE₃_whole[:],seriestype=:scatter, markersize=0.5, markershape=:circle,label="Displacement in Coord 3 ",legendfontsize=7,tickfontsize=9,guidefontsize=9,xlabel="Displacement from ML prediction",ylabel="Displacement from FE")
savefig("R2_Coord3_corrected_V3.pdf")
plot(log10.(Losses), linewidth=3,label="", xlabel="Nº Iterations",ylabel="Loss values",legendfontsize=8,tickfontsize=9,guidefontsize=9)
savefig("Loss_corrected_V3.pdf")

# function sort_and_apply_indices(original_arr, apply_arr)
#     # Create a copy of the original array
#     sorted_arr = copy(original_arr)
    
#     # Sort the array in ascending order
#     sort!(sorted_arr)
    
#     # Find the indices that were changed
#     indices_changed = sortperm(original_arr)
    
#     # Apply the indices to another array
#     result_arr = similar(apply_arr, length(apply_arr))
#     for (i, idx) in enumerate(indices_changed)
#         result_arr[i] = apply_arr[idx]
#     end
    
#     return sorted_arr, indices_changed, result_arr
# end

# #Coord1_y_predicted = vec(y_predicted[1:10,:])
# Coord1_y_predicted = vec(y_predicted)
# Coord2_y_predicted = vec(y_predicted[11:20,:])
# #Coord1_y_fromFE = vec(y_fromFE[1:10,:])
# Coord1_y_fromFE = vec(y_fromFE)
# Coord2_y_fromFE = vec(y_fromFE[11:20,:])

# sorted_Coord1_y_fromFE, indices_Coord1_y_fromFE, sorted_Coord1_y_predicted = sort_and_apply_indices(Coord1_y_fromFE, Coord1_y_predicted)
# sorted_Coord2_y_fromFE, indices_Coord2_y_fromFE, sorted_Coord2_y_predicted = sort_and_apply_indices(Coord2_y_fromFE, Coord2_y_predicted)
# # Coordinate 1 values
# random_indices = rand(1:8000000,400000)

# plot!([sorted_Coord1_y_fromFE[1],sorted_Coord1_y_fromFE[end]],[sorted_Coord1_y_predicted[1],sorted_Coord1_y_predicted[end]],label="R2",linestyle=:dash,linewidth=4)
# plot(sorted_Coord1_y_fromFE[random_indices],sorted_Coord1_y_predicted[random_indices],seriestype=:scatter, markersize=0.5, markershape=:circle,label="Displacement in Coord1",legendfontsize=7,tickfontsize=9,guidefontsize=9,xlabel="Displacement from FE",ylabel="Displacement from ML prediction")
# plot(sorted_Coord1_y_fromFE,sorted_Coord1_y_predicted,seriestype=:scatter, markersize=0.5, markershape=:circle,label="Displacement in Coord1",legendfontsize=7,tickfontsize=9,guidefontsize=9,xlabel="Displacement from FE",ylabel="Displacement from ML prediction")
# savefig("R2_Coord1_corrected_V2.pdf")
# # Coordinate 3 values
# plot!([sorted_Coord2_y_fromFE[1],sorted_Coord2_y_fromFE[end]],[sorted_Coord2_y_predicted[1],sorted_Coord2_y_predicted[end]],label="R2",linestyle=:dash,linewidth=4)
# plot(sorted_Coord2_y_fromFE[random_indices],sorted_Coord2_y_predicted[random_indices],seriestype=:scatter, markersize=0.5, markershape=:circle,label="Displacement in Coord3",legendfontsize=7,tickfontsize=9,guidefontsize=9,xlabel="Displacement from FE",ylabel="Displacement from ML prediction")
# savefig("R2_Coord3_corrected_V2.pdf")
# plot(log10.(Losses), linewidth=3,label="", xlabel="Nº Iterations",ylabel="Loss values",legendfontsize=8,tickfontsize=9,guidefontsize=9)
# savefig("Loss_corrected_V2.pdf")



# -------------------------------------
# Let's plot the trajectory of a point
# -------------------------------------

function denormalise(row::Vector,original_array)
    min = minimum(original_array)
    max = maximum(original_array)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,min + (max-min)*row[i])
    end
  return scaled
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


R2_test = R2Function(vec(y_fromFE),vec(y_predicted))# Just to check that we are importing and treating the data properly. Since the point is from the training, the R2 should be high

sorted_indices = sortperm(eachrow(x_train_subset))

y_fromFE_sorted = y_fromFE[:,sorted_indices]
y_predicted_sorted = y_predicted[:,sorted_indices]


Coord1_y_from_FE_sorted_point = y_fromFE_sorted[1:200,:]
Coord3_y_from_FE_sorted_point = y_fromFE_sorted[201:400,:]
Coord1_y_predicted_sorted_point = y_predicted_sorted[1:200,:]
Coord3_y_predicted_sorted_point = y_predicted_sorted[201:400,:]

# Plot per coordinate
plot(Coord3_y_from_FE_sorted_point[1:end],seriestype=:scatter,markersize=2)
plot!(Coord3_y_predicted_sorted_point[1:end],seriestype=:scatter,markersize=2)

function denormalise(row::Vector,original_array)
    min = minimum(original_array)
    max = maximum(original_array)
    scaled = []
    for i in range(1,size(row,1))
        scaled  = append!(scaled,min + (max-min)*row[i])
    end
  return scaled
end

# We are selecting only 1 point, the first one, which corresponds to Node 41 out of 133 of the face
Coord1_y_predicted_sorted_point_descaled = denormalise(Coord1_y_predicted_sorted_point[1,:],y_train₁_whole[:])
Coord3_y_predicted_sorted_point_descaled = denormalise(Coord3_y_predicted_sorted_point[1,:],y_train₃_whole[:])
Coord1_y_fromFE_sorted_point_descaled = denormalise(Coord1_y_from_FE_sorted_point[1,:],y_train₁_whole[:])
Coord3_y_fromFE_sorted_point_descaled = denormalise(Coord3_y_from_FE_sorted_point[1,:],y_train₃_whole[:])

mat_coords = readdlm("simple_mat_coords.txt")
mat_coords_reshape = reshape(mat_coords,3,266)
Point_211_MatCoords = mat_coords_reshape[:,211] # The node 211 corresponds to the first node in the nodes_indices

writedlm("PlottingTrajectoryParaview_corrected_Rog_V3.csv", hcat(Coord1_y_fromFE_sorted_point_descaled.+Point_211_MatCoords[1],zeros(480).+Point_211_MatCoords[2],Coord3_y_fromFE_sorted_point_descaled.+Point_211_MatCoords[3]),",")
writedlm("PlottingTrajectoryParaview_PRED_corrected_Rog_V3.csv", hcat(Coord1_y_predicted_sorted_point_descaled.+Point_211_MatCoords[1],zeros(480).+Point_211_MatCoords[2],Coord3_y_predicted_sorted_point_descaled.+Point_211_MatCoords[3]),",")
