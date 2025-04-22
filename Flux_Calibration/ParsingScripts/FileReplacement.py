with open('filenames_output_forFE_["0.018", "0.006", "0.222", "0.258", "0.15", "0.174", "0.234", "0.198", "0.3", "0.15", "0.114", "0.174", "0.222", "0.066", "0.234", "0.27", "0.018", "0.282", "0.09", "0.222"].txt', 'r') as file:
    lines = file.readlines()

with open('filenames_output_parsed_forFE_["0.018", "0.006", "0.222", "0.258", "0.15", "0.174", "0.234", "0.198", "0.3", "0.15", "0.114", "0.174", "0.222", "0.066", "0.234", "0.27", "0.018", "0.282", "0.09", "0.222"].txt', 'w') as file:
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        line = line.replace('[', '')  # Remove opening square bracket
        line = line.replace(']', '')  # Remove closing square bracket
        line = line.replace('"', '')  # Remove double quotes
        line = line.replace(",", "")  # Remove commas
        line = line.replace("_"," ")  # Replace underscores with space
        file.write(line + '\n')  # Write modified line to output file