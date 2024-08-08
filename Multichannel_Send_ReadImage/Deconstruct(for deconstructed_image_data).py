#Deconstruction "deconstructed_image_data.txt"

# This file is used to deconstruct an image into 3 rows of data, where each pixel value is divided by 100
# and saved to a text file. The image is first converted to grayscale and resized to 99x99 pixels.

import numpy as np
from PIL import Image

def deconstruct_image(file_path, output_file_path):
    # Load the image
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    gray_scale_img = np.array(img)

    # Ensure the image is 99x99
    Height = 99
    Width = 99
    if gray_scale_img.shape != (Height, Width):  # Height x Width
        raise ValueError(f"Image size must be {Height}x{Width}, but got {gray_scale_img.shape}")

    # Print original image shape
    print(f"Original image shape: {gray_scale_img.shape}")

    # Flatten the image
    vector = gray_scale_img.flatten()

    # Calculate the size of each row (must be divisible by 3)
    total_pixels = Height * Width
    row_size = total_pixels // 3

    # Reshape into 3 rows
    three_rows = vector.reshape(3, row_size)

    print(f"Three rows shape: {three_rows.shape}")

    # Divide the values by 100
    three_rows_divided = three_rows / 100.0

    # Save the three rows to a text file
    np.savetxt(output_file_path, three_rows_divided.T, fmt='%.2f', delimiter='\t')
    print(f"Deconstructed image data saved to {output_file_path}")

if __name__ == "__main__":
    file_path = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\send image\sizephoto.jpg"
    output_file_path = r"C:\Users\bovta\Desktop\Aaron (Intern)\Aaron (Intern)\VS code Stuff\send image\deconstructed_image_data.txt"
    
    deconstruct_image(file_path, output_file_path)