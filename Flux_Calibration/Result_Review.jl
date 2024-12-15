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
directory_path = "NN_parametric_run_corrected_V3/"
file_names, R2, File = list_files_in_directory(directory_path)




#----------------------------------------------------------
# With the model trained, let's import the data and compare
#----------------------------------------------------------

#TODO Double check with Rogelio; we are computing the R2 of the evaluation only on the nodes that we chose; it's difficult to reconstruct from that because the nodes are chosen randomly and we don't have those saved
