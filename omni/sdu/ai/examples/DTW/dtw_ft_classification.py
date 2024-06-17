# This code implements a Fault detection algorithm using dynamic time warping (DTW) and
# discriminant analysis for force-torque and position-orientation assembly data

import pandas as pd
import os
import numpy as np
import ast  # Library to safely evaluate literal expressions from strings
from fastdtw import fastdtw


data_directory = "../../../../../Documents/sdu.extensions.hotfix/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv"

# List to store dataframes from each CSV file
dataframes = []

# Loop through CSV files in the directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_directory, filename)
        # Read the CSV file into a list of lists
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Check if the last line contains a valid label
        try:
            label = int(lines[-1].strip())
        except ValueError:
            print(f"Error: Invalid label in file {filename}")
            continue

        # Convert each line to a list using ast.literal_eval
        data = []
        for line in lines[:-1]:  # Exclude the last line
            try:
                # Split the line into forces and positions
                forces_str, positions_str = line.strip().split('\t')
                forces = ast.literal_eval(forces_str)
                positions = ast.literal_eval(positions_str)
                data.append((forces, positions))
            except (SyntaxError, ValueError, TypeError) as e:
                print(f"Error: Unable to process line {line} in file {filename}: {e}")

        # Extract forces and positions
        forces = [item[0][0] for item in data]
        positions = [item[1][0] for item in data]
        # Extract the label from the last line
        label = int(lines[-1].strip())  # Assuming the label is an integer
        # Create a DataFrame for each file
        file_df = pd.DataFrame({'Forces': forces, 'Positions': positions, 'Label': label})
        # Append the DataFrame to the list
        dataframes.append(file_df)

# Concatenate dataframes into one
full_df = pd.concat(dataframes, ignore_index=True)
print(full_df)
# print(full_df.iloc[1])

# breakpoint()

# Combine 'Forces' and 'Positions' into a single feature column 'Features'
full_df['Features'] = full_df.apply(lambda row: (row[0], row[1]), axis=1)
print(full_df['Features'][0])
print("len(full_df['Features'][0]): ", len(full_df['Features'][0]))

# Split the data into features (X) and target (y)
X = full_df['Features']
y = full_df['Label']

# Assuming 'Features' column contains tuples of Forces and Positions
full_df['Force'] = full_df['Features'].apply(lambda x: x[0])  # Assuming Forces are at index 0 in the tuple
full_df['Position'] = full_df['Features'].apply(lambda x: x[1])  # Assuming Positions are at index 1 in the tuple

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Force'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

# For this approach I will use only the force in the Z axis:
# Extract 'Force_3' time series for a reference curve
reference_curve = force_df['Force_3'][0]

# Loop through the rest of the curves and calculate DTW distances
for index, row in full_df.iterrows():
    new_curve = row['Force_3']

    # Compute DTW distance
    distance, path = fastdtw(reference_curve, new_curve)

    # Set a threshold (you may need to determine an appropriate threshold based on your data)
    threshold = 10  # Example threshold, adjust according to your needs

    # Print or use the distance for further analysis
    print(f"DTW Distance between reference and curve at index {index}: {distance}")

    # Compare the distance with the threshold for failure detection
    if distance > threshold:
        print(f"Potential Failure Detected for curve at index {index}!")
    else:
        print(f"No Failure Detected for curve at index {index}.")