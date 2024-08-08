#Deconstruction for 255 values "deconstructed_array_data.txt"

# This file is used to create a 1D array with values from 0 to 255, reshape it into 3 rows, divide the values by 100, and save the three rows to a text file.
# This is a test file to see if the deconstruction works for the 255 values in the "deconstructed_array_data.txt" file.
import numpy as np

def create_and_deconstruct_array(output_file_path):
    # Create a 1D array with values from 0 to 255
    vector = np.arange(256, dtype=np.float64)

    # Print original array shape
    print(f"Original array shape: {vector.shape}")

    # Calculate the size of each row (must be divisible by 3)
    total_elements = len(vector)
    row_size = total_elements // 3

    # If the total number of elements is not divisible by 3, we'll pad the array
    if total_elements % 3 != 0:
        padding = 3 - (total_elements % 3)
        vector = np.pad(vector, (0, padding), mode='constant', constant_values=0)
        total_elements = len(vector)
        row_size = total_elements // 3

    # Reshape into 3 rows
    three_rows = vector.reshape(3, row_size)

    print(f"Three rows shape: {three_rows.shape}")

    # Divide the values by 100
    three_rows_divided = three_rows / 100.0

    # Save the three rows to a text file
    np.savetxt(output_file_path, three_rows_divided.T, fmt='%.2f', delimiter='\t')
    print(f"Deconstructed array data saved to {output_file_path}")

if __name__ == "__main__":
    output_file_path = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\send image\deconstructed_array_data.txt"
    
    create_and_deconstruct_array(output_file_path)