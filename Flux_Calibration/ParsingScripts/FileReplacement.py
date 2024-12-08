with open('filenames_output_complex_potential.txt', 'r') as file:
    lines = file.readlines()

with open('filenames_parsed_complex_potential.txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        line = line.replace('[', '')  # Remove opening square bracket
        line = line.replace(']', '')  # Remove closing square bracket
        line = line.replace('"', '')  # Remove double quotes
        line = line.replace(",", "")  # Remove commas
        line = line.replace("_"," ")  # Replace underscores with space
        file.write(line + '\n')  # Write modified line to output file