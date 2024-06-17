import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
import numpy as np
import ast  # Library to safely evaluate literal expressions from strings

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from yellowbrick.classifier import ClassificationReport
import torch.nn.functional as F

from tqdm import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


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

# Combine Force and Position DataFrames
X_numeric = pd.concat([force_df, position_df], axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)  # The features are transformed to have a mean of 0 and a standard deviation of 1
print(X_scaled[0])
print("len(X_scaled[0]): ", len(X_scaled[0]))

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y.values).long()  # Assuming y is a Pandas Series with integer labels

def reshape_for_lstm(X, y, sequence_length):
    num_samples, num_features = X.shape
    num_batches = num_samples - sequence_length + 1

    # Create empty arrays to store batches
    sequences = np.zeros((num_batches, sequence_length, num_features))
    targets = y[sequence_length - 1:]

    # Fill in the batches
    for i in range(num_batches):
        sequences[i] = X[i:i + sequence_length]

    return sequences, targets

# Define sequence length
sequence_length = 400  # You can adjust this based on your requirement

# Reshape for LSTM
X_reshaped, y_reshaped = reshape_for_lstm(X_tensor.numpy(), y_tensor.numpy(), sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

# Instantiate the model
# Instantiate the model
input_size = X_train.shape[2]
hidden_size = 16
num_layers = 1
num_classes = 2  # Adjusted to 2 for binary classification

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Defines an Adam optimizer to update the model parameters during training.

batch_size = 2  # Adjust this to a smaller value
X_train_tensor = X_train_tensor.view(-1, sequence_length, input_size)  # Reshape the input
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

accumulation_steps = 4  # Accumulate gradients over 4 steps
epochs = 50
# Training loop
for epoch in tqdm(range(epochs), desc="Training"):
    optimizer.zero_grad()
    for i in range(0, len(train_loader), accumulation_steps):
        batch_X, batch_y = next(iter(train_loader))
        output = model(batch_X)
        
        # Assuming classes are 0 or 1, use BCEWithLogitsLoss
        loss = F.cross_entropy(output, batch_y)

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Create a test dataset and loader
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Evaluation loop
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    predictions = []

    # Use tqdm for a progress bar
    for batch_X, batch_y in tqdm(test_loader, desc="Evaluating"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())

        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

# Calculate accuracy
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test_tensor.cpu().numpy(), predictions))

# Plot results--------
# Calculate confusion matrix
# cm = confusion_matrix(y_test_flat, binary_predictions_flat)

# # Plot confusion matrix as a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')

# # Save the figure instead of showing it interactively
# plt.savefig('confusion_matrix.png')