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

from tqdm import tqdm

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
full_df['Features'] = full_df[['Forces', 'Positions']].apply(tuple, axis=1)
print(full_df['Features'][0])
print("len(full_df['Features'][0]): ", len(full_df['Features'][0]))

# Split the data into features (X) and target (y)
X = full_df['Features']
y = full_df['Label']

# Extract numerical values from tuples and flatten the lists
X_numeric = pd.DataFrame(X.tolist(), columns=['Forces', 'Positions'])
X_numeric['Forces'] = X_numeric['Forces'].apply(lambda x: x[0])
X_numeric['Positions'] = X_numeric['Positions'].apply(lambda x: x[0])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric) # the features are transformed in a way that they have a mean of 0 and a standard deviation of 1
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
input_size = X_train.shape[2]
hidden_size = 64  # You can adjust this. before 128
num_layers = 1  # You can adjust this. before: 2
num_classes = 2  # Assuming 2 classes
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# Move model to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# criterion = nn.BCELoss() # Creates a binary cross-entropy loss function. This is often used for binary classification problems.
optimizer = optim.Adam(model.parameters(), lr=0.001) # Defines an Adam optimizer to update the model parameters during training.

# Training loop
# for epoch in tqdm(range(epochs), desc="Training"):
#     optimizer.zero_grad()
#     for i in range(0, len(train_loader), accumulation_steps):
#         batch_X, batch_y = next(iter(train_loader))
#         output = model(batch_X)
#         loss = criterion(output, batch_y)
#         loss.backward()

#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        
# Training loop
epochs = 100
for epoch in tqdm(range(epochs), desc='Training Progress', unit='epoch'):
    optimizer.zero_grad() # Zeroes the gradients of the model parameters to avoid accumulation.
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor) # Calculates the loss using the specified criterion (binary cross-entropy in this case).
    loss.backward()
    optimizer.step() # Updates the model parameters using the optimizer

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions on the testing set
with torch.no_grad():
    # Move the model to the CPU
    model.to("cpu")
    # Move the input tensor to the CPU before making predictions
    X_test_cpu = X_train.to("cpu")
    predictions = model(X_test_cpu).squeeze().numpy()

# Save the trained model
torch.save(model.state_dict(), '/home/asegui/Documents/sdu.extensions.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model/modelv3.pth')

# Convert probabilities to binary predictions
binary_predictions = (predictions >= 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, binary_predictions)
classification_report_str = classification_report(y_test, binary_predictions)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)

# Plot results--------
# Convert probabilities to binary predictions
binary_predictions = (predictions >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, binary_predictions)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Assuming `y_test` is the true labels and `binary_predictions` are the predicted labels
# Calculate the classification report
report = classification_report(y_test, binary_predictions, output_dict=True)

# Convert the classification report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f', linewidths=.5, cbar=False)
plt.title('Classification Report Heatmap')
plt.show()
