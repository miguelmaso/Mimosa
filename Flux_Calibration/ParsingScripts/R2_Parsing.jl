using JSON
using Glob

# Function to parse the filename
function parse_filename(filename)
    # Extract numbers from the filename
    regex = r"Layers:(\d+) Neurons:(\d+) Experiments:(\d+) Nodes:(\d+)Iter:(\d+) Corrected"
    matching = match(regex, filename)
    
    if matching !== nothing
        # Convert the extracted strings to numbers
        layers = parse(Int, matching.captures[1])
        neurons = parse(Int, matching.captures[2])
        experiments = parse(Int, matching.captures[3])
        nodes = parse(Int, matching.captures[4])
        iterations = parse(Int, matching.captures[5])
        return (layers, neurons, experiments, nodes, iterations)
    else
        return nothing  # if the filename doesn't matching the pattern
    end
end

# Function to filter files based on experiments, nodes, and iterations
function filter_files(experiments, nodes, iterations, file_path_pattern)
    # Get list of all json files matchinging the file_path_pattern
    files = glob(file_path_pattern, ".")
    
    selected_files = []
    for file in files
        filename = basename(file)
        parsed = parse_filename(filename)
        if parsed !== nothing
            _, _, exp, node, iter = parsed
            # Check if the file matchinges the input conditions
            if exp == experiments && node == nodes && iter == iterations
                push!(selected_files, file)
            end
        end
    end
    
    return selected_files
end

# Function to extract the "R2" value from selected JSON files
function extract_R2_values(files)
    R2_values = Dict()
    
    for file in files
        # Read and parse JSON content
        json_content = JSON.parsefile(file)
        if haskey(json_content, "R2")
            R2_values[file] = json_content["R2"]
        else
            println("No 'R2' key found in file: $file")
        end
    end
    
    return R2_values
end

# Main function to call
function get_R2_values_for_conditions(experiments::Int, nodes::Int, iterations::Int)
    # Adjust the file path pattern if needed
    file_path_pattern = "*.json"
    
    # Filter the files
    selected_files = filter_files(experiments, nodes, iterations, file_path_pattern)
    
    if length(selected_files) == 0
        println("No files found matching the criteria.")
        return
    end
    
    # Extract R2 values
    R2_values = extract_R2_values(selected_files)
    
    # Display the R2 values
    println("R2 values:")
    for (file, R2) in R2_values
        println("$file : $R2")
    end
end

# Example usage: input the desired experiments, nodes, and iterations
get_R2_values_for_conditions(10000, 200, 10000)
