with open('filenames_output_FE_Trajectory.txt', 'r') as file:
    lines = file.readlines()

with open('filenames_parsed_FE_Trajectory.txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        line = line.replace('[', '')  # Remove opening square bracket
        line = line.replace(']', '')  # Remove closing square bracket
        line = line.replace('"', '')  # Remove double quotes
        line = line.replace(",", "")  # Remove commas
        file.write(line + '\n')  # Write modified line to output file