import os

# Define the directory where the .en files are located
directory_path = '../manual_data/english'  # This is where your individual .en files are located

# Define the name and path for the combined file
combined_file_path = '../manual_data/english/combined.en'  # Save combined file in the same directory

# Open the combined file in write mode
with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a .en extension
        if filename.endswith('.en'):
            file_path = os.path.join(directory_path, filename)
            # Open and read the content of the .en file
            with open(file_path, 'r', encoding='utf-8') as file:
                # Write the content to the combined file
                combined_file.write(file.read())
                # Add a newline after each file content (optional)
                combined_file.write('\n')

print(f"All .en files have been combined into {combined_file_path}")
