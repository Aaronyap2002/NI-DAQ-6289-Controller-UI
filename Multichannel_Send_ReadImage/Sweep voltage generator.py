# Generate sweeping voltage
import numpy as np

# Generate the voltage sweep
start_voltage = 0
end_voltage = 5
num_samples = 1000

voltages = np.linspace(start_voltage, end_voltage, num_samples)

# Create a file and write the voltages
filename = "voltage_sweep.txt"

with open(filename, 'w') as file:
    for voltage in voltages:
        file.write(f"{voltage:.6f}\n")

print(f"Voltage sweep from {start_voltage}V to {end_voltage}V with {num_samples} samples has been written to {filename}")

# Optional: Print the first few and last few values to verify
print("\nFirst 5 values:")
print(voltages[:5])
print("\nLast 5 values:")
print(voltages[-5:])