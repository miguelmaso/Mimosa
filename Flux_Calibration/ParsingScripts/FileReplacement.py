with open('filenames_output_forFE_["0.246", "0.0", "0.09", "0.102"].txt', 'r') as file:
    lines = file.readlines()

with open('filenames_output_parsed_forFE_["0.246", "0.0", "0.09", "0.102"].txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        line = line.replace('[', '')  # Remove opening square bracket
        line = line.replace(']', '')  # Remove closing square bracket
        line = line.replace('"', '')  # Remove double quotes
        line = line.replace(",", "")  # Remove commas
        line = line.replace("_"," ")  # Replace underscores with space
        file.write(line + '\n')  # Write modified line to output file