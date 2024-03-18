using Flux
using Flux: train!
using Statistics
using Plots
using Zygote
using DelimitedFiles
#-------------------------------------------------------------------------------
# Test and training data
#-------------------------------------------------------------------------------
x_train = readdlm("Input_Alberto_.txt", ',');
y_train = readdlm("Output_Alberto_.txt", ',');

x_train = x_train'
y_train = y_train'

function sort_with_indices(arr)
    # Create a copy of the original array
    sorted_arr = copy(arr)
    
    # Sort the array in ascending order
    sort!(sorted_arr, dims=2)
    
    # Find the indices that were changed
    indices_changed = findall(x -> arr[x] != sorted_arr[x], 1:length(arr))
    
    return sorted_arr, indices_changed
end


sorted_y_train, indices = sort_with_indices(y_train)


#plot(y_train, seriestype = :scatter, label=["Original Data"])
