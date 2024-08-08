# This python file is used to determine the gradient of the power vs voltage graph of the photodiode based on the settings (gain) of the photodiode. 
# This plots takes in up to 12 excels files and plots the power vs voltage graph of the photodiode.
# The values of each settings (gain) are stored in an excel file
# Plot is displayed as "newplot.png" file

import pandas as pd
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx *.xls")], 
                                             title="Select Excel files")
    if file_paths:
        create_plot(file_paths)

def calculate_gradients(x, y):
    gradients = np.diff(y) / np.diff(x)
    gradients = np.append(gradients, gradients[-1])
    return gradients

def create_plot(file_paths):
    fig = go.Figure()

    # Updated color palette with more distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_excel(file_path, header=None, names=['Power (µW)', 'Voltage (V)'])
        
        x = df['Power (µW)'].values
        y = df['Voltage (V)'].values
        gradients = calculate_gradients(x, y)

        # Calculate best-fit line
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Create points for best-fit line, including y-intercept
        x_fit = np.array([0, np.max(x)])
        y_fit = slope * x_fit + intercept

        # Cycle through colors if more than 12 files
        color = colors[i % len(colors)]

        # Add scatter plot for Power vs Voltage
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name=f'Data - {file_name}',
                marker=dict(color=color),
                hovertemplate='Power: %{x} µW<br>Voltage: %{y} V<br>Gradient: %{text} V/µW<extra></extra>',
                text=[f'{g:.8e}' for g in gradients]
            )
        )

        # Add best-fit line
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'Fit - {file_name}',
                line=dict(color=color, dash='dash'),
                hovertemplate='Power: %{x} µW<br>Voltage: %{y} V<extra></extra>'
            )
        )

        # Add text annotations
        fig.add_annotation(
            x=1, y=1-i*0.04, xref="paper", yref="paper", xanchor="right", yanchor="top",
            text=f"{file_name}<br>Slope: {slope:.4e} V/µW, Y-Int: {intercept:.4e} V",
            showarrow=False, font=dict(size=9, color=color), bgcolor="white", 
            bordercolor=color, borderwidth=1
        )

    fig.update_layout(
        title='Power vs Voltage - Multiple Datasets',
        xaxis_title='Power (µW)',
        yaxis_title='Voltage (V)',
        hovermode='closest',
        height=900,
        width=1200,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=8)
        ),
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white'  # Set paper background to white
    )

    fig.show()

# Create the main window
root = tk.Tk()
root.title("Power vs Voltage Plotter")

# Create and pack a button
upload_button = tk.Button(root, text="Upload Excel Files (max12)", command=select_files)
upload_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
