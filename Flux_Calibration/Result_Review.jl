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

