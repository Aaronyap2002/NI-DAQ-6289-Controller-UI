# This file is used to send data to the NI DAQ and read the data back. The data is processed through a look-up table before being sent to the DAQ. The acquired data is saved to files 'After_LUT_v.txt' and 'Read_Pd.txt'.
# The number of samples per channel is determined by the number of columns in the input file 'deconstructed_image_data.txt'.
# The look-up table is read from 'Look-up table(1x10pwr2).txt'. 
# After_LUT_v.txt: Compare the deconstructed pixel value with the 2nd column of the LUT and find the closest value in the LUT. Replace the pixel value with the corresponding value from the 1st column of the LUT.
# After_LUT_v.txt: The processed data that will be sent to the DAQ.
# Read_Pd.txt: The acquired data from the DAQ which will be used to reconstruct the image.

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter

def prepare_multichannel_data(file_path, num_channels):
    data = np.loadtxt(file_path, delimiter='\t')
    if data.shape[1] < num_channels:
        padding = np.zeros((data.shape[0], num_channels - data.shape[1]))
        data = np.hstack((data, padding))
    elif data.shape[1] > num_channels:
        data = data[:, :num_channels]
    return data.T

def read_lookup_table(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    input_values = data[:, 0]
    output_values = data[:, 1]
    return input_values, output_values

def find_closest_value(value, array):
    return array[np.argmin(np.abs(array - value))]

# Constants
SAMPLE_RATE = 100
NUM_CHANNELS = 3

# Prepare write data from file
file_path = 'deconstructed_image_data.txt'
write_data = prepare_multichannel_data(file_path, NUM_CHANNELS)
SAMPLES_PER_CHANNEL = write_data.shape[1]

# Read the look-up table
input_lut, output_lut = read_lookup_table('Look-up table(1x10pwr2).txt')

# Process write_data through look-up table
After_LUT_v = np.zeros_like(write_data)
for channel in range(NUM_CHANNELS):
    for i in range(SAMPLES_PER_CHANNEL):
        closest_output = find_closest_value(write_data[channel][i], output_lut)
        corresponding_input = input_lut[np.where(output_lut == closest_output)[0][0]]
        After_LUT_v[channel][i] = corresponding_input

# Ensure After_LUT_v is C-contiguous
After_LUT_v = np.ascontiguousarray(After_LUT_v)

with nidaqmx.Task() as read_task, nidaqmx.Task() as write_task:
    # Setup channels
    for i in range(NUM_CHANNELS):
        read_task.ai_channels.add_ai_voltage_chan(f"Dev1/ai{i}", terminal_config=TerminalConfiguration.RSE, min_val=-10.0, max_val=10.0)
        write_task.ao_channels.add_ao_voltage_chan(f"Dev1/ao{i}", min_val=-10.0, max_val=10.0)

    # Configure timing for both tasks
    write_task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=SAMPLES_PER_CHANNEL)
    read_task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE, source="ao/SampleClock", sample_mode=AcquisitionType.FINITE, samps_per_chan=SAMPLES_PER_CHANNEL)

    # Configure start trigger for read task
    read_task.triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/ao/StartTrigger")
    
    read_task.timing.delay_from_samp_clk_delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
    read_task.timing.delay_from_samp_clk_delay = 0.5/SAMPLE_RATE # update this with the value in seconds to delay

    # Create reader and writer
    reader = AnalogMultiChannelReader(read_task.in_stream)
    writer = AnalogMultiChannelWriter(write_task.out_stream)

    # Write After_LUT_v data
    writer.write_many_sample(After_LUT_v)
    
    # Start the tasks
    read_task.start()
    write_task.start()
    
    # Read the acquired data
    read_data = np.zeros((NUM_CHANNELS, SAMPLES_PER_CHANNEL), dtype=np.float64)
    reader.read_many_sample(read_data, number_of_samples_per_channel=SAMPLES_PER_CHANNEL, timeout=SAMPLES_PER_CHANNEL/SAMPLE_RATE + 5.0)
    
    # Wait for the tasks to complete
    write_task.wait_until_done()
    read_task.wait_until_done()
    
    # Stop the tasks
    write_task.stop()
    read_task.stop()

# Save After_LUT_v values to file
np.savetxt('After_LUT_v.txt', After_LUT_v.T, delimiter='\t', fmt='%.6f')

# Save Read_Pd (read_data) values to file
np.savetxt('Read_Pd.txt', read_data.T, delimiter='\t', fmt='%.6f')

print(f"Data acquisition and processing complete. Results written to 'After_LUT_v.txt' and 'Read_Pd.txt'.")
print(f"Number of samples per channel: {SAMPLES_PER_CHANNEL}")