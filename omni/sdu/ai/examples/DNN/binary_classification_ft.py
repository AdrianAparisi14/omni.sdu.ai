import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Sample Dataset Class
class ForceTorqueDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Sample Neural Network Model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # Assuming binary classification

    def forward(self, x):
        return self.fc(x)

# Sample Data Generation
np.random.seed(42)
num_samples = 1000
force_torque_data = np.random.rand(num_samples, 6)  # Assuming 6-dimensional force-torque measurements
labels = np.random.randint(2, size=num_samples)  # Binary labels: 0 or 1

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(force_torque_data, labels, test_size=0.2, random_state=42)

# Create DataLoader for training
train_dataset = ForceTorqueDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
input_size = force_torque_data.shape[1]
model = SimpleClassifier(input_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation on the test set
test_dataset = ForceTorqueDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
