with open('filenames_output.txt', 'r') as file:
    lines = file.readlines()

with open('filenames_parsed.txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        line = line.replace('[', '')  # Remove opening square bracket
        line = line.replace(']', '')  # Remove closing square bracket
        line = line.replace('"', '')  # Remove double quotes
        file.write(line + '\n')  # Write modified line to output file
