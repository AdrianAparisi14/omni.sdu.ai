from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
import numpy as np
import ast  # Library to safely evaluate literal expressions from strings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from yellowbrick.classifier import ClassificationReport
import torch.nn.functional as F

from tqdm import tqdm

import os

# Directory containing your CSV files
data_directory = '../../../../../Documents/sdu.extensions.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/csv'

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
position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])



# Assuming force_df is your feature matrix and labels_df is your target vector
# Concatenate Force_3 column with other relevant features if necessary
X = np.array(force_df['Force_3'].tolist())
y = np.array(labels_df['Label'].tolist())

# Reshape X to have dimensions (number of assemblies, 400, 1)
X = X.reshape((len(X) // 400, 400, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# LSTM Model
model = Sequential()
model.add(LSTM(50, input_shape=(400, 1)))
model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (Success or Failure)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation
print("LSTM Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Optionally, save the trained model
# torch.save(model.state_dict(), '../../../../../Documents/sdu.extensions.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model/modelv7.pth')

# Model Inference
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)

# Convert the continuous output to a binary value (0 or 1) using softmax activation
_, predictions = torch.max(outputs, 1)

# Calculate Accuracy
accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
report = classification_report(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
print("Classification Report:")
print(report)