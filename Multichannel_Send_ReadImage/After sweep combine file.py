# Combine 2 text file to prepare for lookup table for a particular setting.

# Left column: voltage_sweep.txt
# Right column: 1x10pwr2.txt (voltage sweep performed with this gain setting)

import numpy as np

def combine_channel_files(file_paths, output_file, num_channels):
    # Read data from each file
    channel_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            # Skip header if present
            first_line = f.readline().strip()
            if not is_float(first_line.split(',')[0]):  # Check if the first item is not a float
                print(f"Skipping header in {file_path}: {first_line}")
            else:
                f.seek(0)  # If no header, go back to the start of the file
            
            # Read data, handling both comma-separated and space/tab-separated formats
            data = []
            for line in f:
                try:
                    # Try comma-separated first
                    values = [float(val.strip()) for val in line.split(',') if val.strip()]
                    if len(values) == 0:
                        continue
                    if len(values) > 1:
                        data.append(values[-1])  # Take the last value if multiple columns
                    else:
                        data.append(values[0])
                except ValueError:
                    # If comma-separated fails, try space/tab-separated
                    try:
                        values = [float(val.strip()) for val in line.split() if val.strip()]
                        if len(values) > 0:
                            data.append(values[-1])  # Take the last value if multiple columns
                    except ValueError:
                        print(f"Skipping invalid line in {file_path}: {line.strip()}")
            
            channel_data.append(np.array(data))
    
    # Ensure all channels have the same number of samples
    min_samples = min(len(data) for data in channel_data)
    channel_data = [data[:min_samples] for data in channel_data]
    
    # Pad with zeros if we have fewer channels than specified
    while len(channel_data) < num_channels:
        channel_data.append(np.zeros(min_samples))
    
    # Combine data into a 2D array
    combined_data = np.column_stack(channel_data[:num_channels])
    
    # Save combined data to a new text file
    with open(output_file, 'w') as f:
        f.write("VOA Voltage\tPhotodiode Voltage\n")  # Write header
        for row in combined_data:
            f.write('\t'.join(f'{val:.6f}' for val in row) + '\n')
    
    print(f"Combined data saved to {output_file}")
    print(f"Shape of combined data: {combined_data.shape}")

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Usage
file_paths = [
    'voltage_sweep.txt',
    '1x10pwr2.txt'
]
output_file = 'Look-up table(1x10pwr2).txt'
NUM_CHANNELS = 2  # Changed to 2 to include both input files

combine_channel_files(file_paths, output_file, NUM_CHANNELS)
