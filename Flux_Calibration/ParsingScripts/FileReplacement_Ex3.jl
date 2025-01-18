using DelimitedFiles
cd("/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/ParsingScripts/")
Potential_Input = readdlm("LHS_Conf3.txt")


# Define the directory containing the .txt files
data_dir = "/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/ParsingScripts/Configuration3"

# List all .txt files in the directory
txt_files = filter(file -> endswith(file, ".txt"), readdir(data_dir))

# Define an array to hold parsed file information
parsed_files = Vector{Tuple{Int, Float64, String}}()  # Ensure specific type

# Parse and store file information
for file in txt_files
    match = Base.match(r"^(\d+)___([\d\.]+)\.txt$", file)
    if match !== nothing
        combo_index = parse(Int, match.captures[1])
        load_factor = parse(Float64, match.captures[2])
        push!(parsed_files, (combo_index, load_factor, file))
    else
        println("Warning: File $file does not match the naming pattern.")
    end
end

# Sort parsed files by combination index (ascending) and load factor (ascending)
# Sort both by combo_index and load_factor in ascending order
sorted_files = sort(parsed_files, by = x -> (x[1], x[2]))

# Initialize matrices
n_files = length(sorted_files)
Scaled_Vectors = Matrix{Float64}(undef, 5, n_files)  # 5xN matrix
File_Contents = Matrix{Float64}(undef, 1755, n_files)  # 1755xN matrix

# Process each file in the sorted order
for (i, (combo_index, load_factor, file)) in enumerate(sorted_files)
    # Extract the corresponding column from Potential_Input
    if combo_index <= size(Potential_Input, 2)
        potential_vector = Potential_Input[:, combo_index]

        # Multiply the vector by the load factor and store in Scaled_Vectors
        Scaled_Vectors[:, i] = potential_vector .* load_factor
    else
        println("Warning: Combo index $combo_index exceeds array size.")
    end

    # Read the contents of the file and store in File_Contents
    file_path = joinpath(data_dir, file)
    file_vector = readdlm(file_path)  # Read file contents as a vector
    if length(file_vector) == 1755
        File_Contents[:, i] = file_vector
    else
        println("Warning: File $file does not contain 1755 components.")
    end
end

# Output the matrices
println("Scaled Vectors Matrix (5x$n_files):")
#println(Scaled_Vectors)

println("File Contents Matrix (1755x$n_files):")
#println(File_Contents)

open("filenames_ex3.txt","w") do io
    writedlm(io,Scaled_Vectors)
end
open("contents_ex3.txt","w") do io
    writedlm(io,File_Contents)
end
