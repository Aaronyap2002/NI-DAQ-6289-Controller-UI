#Reconstruction

# This file is used to reconstruct an image from the 3 rows of data saved in the text file.
# The data is first multiplied by 100 and converted back to uint8 to get the pixel values.
# The 3 rows are then combined to reconstruct the image, which is saved and displayed.

import numpy as np
import matplotlib.pyplot as plt

def reconstruct_image(input_file_path, output_image_path):
    # Load the data from the text file
    three_rows_divided = np.loadtxt(input_file_path, delimiter='\t').T

    # Multiply by 100 and convert back to uint8
    three_rows = (three_rows_divided * 100).astype(np.uint8)

    # Reconstruct the image
    Height = 99
    Width = 99
    reconstructed_vector = three_rows.flatten()
    reconstructed_img = reconstructed_vector.reshape((Height, Width))

    print(f"Reconstructed image shape: {reconstructed_img.shape}")

    # Save the reconstructed image
    plt.imsave(output_image_path, reconstructed_img, cmap='gray')
    print(f"Reconstructed image saved to {output_image_path}")

    # Display the reconstructed image
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title("Reconstructed Image")
    plt.show()

if __name__ == "__main__":
    input_file_path = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\send image\Read_Pd.txt"
    output_image_path = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\send image\reconstructed_image.png"
    
    reconstruct_image(input_file_path, output_image_path)