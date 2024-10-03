import os
import numpy as np

def find_txt_files(base_directory):
    """Find all .txt files in base_directory and its subdirectories, and return their names without extension and their contents."""
    txt_files = []
    contents = []
    
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_name_without_extension = os.path.splitext(file)[0]
                txt_files.append(file_name_without_extension)
                
                with open(file_path, 'r') as f:
                    file_content = f.read().splitlines()
                    contents.append(file_content)
    
    return txt_files, contents

def write_filenames_to_file(filenames, output_file):
    """Write filenames to the output file, one per line."""
    with open(output_file, 'w') as outfile:
        for name in filenames:
            outfile.write(f"{name}\n")

def write_contents_to_file(contents, output_file):
    """Write contents array to the output file."""
    # Convert contents list to a NumPy array and transpose it for the 399xn format
    content_array = np.array(contents).T
    with open(output_file, 'w') as outfile:
        for row in content_array:
            outfile.write("\t".join(row) + "\n")

def main():
    base_directory = '/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/DataEngineering/Complex_Potential ["0.3", "0.042", "0.222", "0.282", "0.246", "0.09", "0.198", "0.126", "0.294", "0.162", "0.018", "0.078", "0.066", "0.114", "0.138", "0.114", "0.294", "0.006", "0.114", "0.21"]'  # Change this to your target directory
    filenames_output_file = 'filenames_output.txt'  # Output file for filenames without extensions
    contents_output_file = 'contents_output.txt'  # Output file for contents array
    
    filenames, contents = find_txt_files(base_directory)
    
    if not contents:
        print("No .txt files found or .txt files are empty.")
        return
    
    # # Ensure all files have exactly 399 entries
    # for i, content in enumerate(contents):
    #     if len(content) != 399:
    #         raise ValueError(f"File {filenames[i]}.txt does not have exactly 399 entries.")
    
    write_filenames_to_file(filenames, filenames_output_file)
    write_contents_to_file(contents, contents_output_file)
    
    print(f"All .txt file names without extension have been written to {filenames_output_file}")
    print(f"Contents of each .txt file have been written to {contents_output_file} in a 399xn format")

if __name__ == "__main__":
    main()
