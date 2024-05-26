import os
import numpy as np
import h5py

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

def save_to_hdf5(filenames, contents, output_file):
    """Save filenames and contents to an HDF5 file."""
    with h5py.File(output_file, 'w') as hdf:
        hdf.create_dataset('filenames', data=[name.encode() for name in filenames])
        
        # Save each file's contents as a separate dataset
        for i, content in enumerate(contents):
            hdf.create_dataset(f'file_{i}', data=np.array(content, dtype='S'))

def main():
    base_directory = '/home/alberto/LINUX_DATA/JuliaRepo/Mimosa/Flux_Calibration/DataResults/'  # Change this to your target directory
    output_file = 'output.h5'  # Output HDF5 file
    
    filenames, contents = find_txt_files(base_directory)
    
    if not contents:
        print("No .txt files found or .txt files are empty.")
        return
    
    # # Ensure all files have exactly 399 entries
    # for i, content in enumerate(contents):
    #     if len(content) != 399:
    #         raise ValueError(f"File {filenames[i]}.txt does not have exactly 399 entries.")
    
    save_to_hdf5(filenames, contents, output_file)
    
    print(f"All .txt file names and their contents have been saved to {output_file}")

if __name__ == "__main__":
    main()
