import os

def write_filenames_and_contents(base_directory, filenames_output_file, contents_output_file):
    """Process files incrementally to minimize memory usage."""
    with open(filenames_output_file, 'w') as filenames_outfile, open(contents_output_file, 'w') as contents_outfile:
        for root, dirs, files in os.walk(base_directory):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    file_name_without_extension = os.path.splitext(file)[0]

                    # Write filename to the output file
                    filenames_outfile.write(f"{file_name_without_extension}\n")

                    with open(file_path, 'r') as f:
                        # Read and write the file contents line by line
                        for line in f:
                            contents_outfile.write(line.strip() + "\t")
                        contents_outfile.write("\n")  # End the row for this file's content

def main():
    # Define base directory and output file paths
    base_directory = '/home/alberto/LINUX_DATA/DESCARGAS/Complex_Potential_Results_Rogelio/Configuration2/'
    filenames_output_file = 'filenames_output_complex_potential.txt'  # Output file for filenames
    contents_output_file = 'contents_output_complex_potential.txt'  # Output file for contents

    # Process the files and write to the output files
    write_filenames_and_contents(base_directory, filenames_output_file, contents_output_file)
    
    print(f"All .txt file names without extension have been written to {filenames_output_file}")
    print(f"Contents of each .txt file have been written to {contents_output_file}")

if __name__ == "__main__":
    main()

