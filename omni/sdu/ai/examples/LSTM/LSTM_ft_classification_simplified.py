    # """This is code provides a LSTM implementation for failure detection on time series of the force recording
    # on the Z axis on the wrist of the UR5. It is tailored to predict either success (1) or failure (0) from an input 
    # time series result of an asembly
    
    # Last update and models generated (18/04/2024): 
    #     - model for real mock novo part (/force_dataset/csv_real_robot_admittance): "/model_real_robot/LSTMmodel2.h5"; "/scaler/scaler_LSTMmodel2.joblib"
    #     - model for real novo part under admittance control (/force_dataset/csv_real_robot_admittance_novo): "/model_real_robot_novo/admittance/LSTMmodel_real_part.h5"; "/scaler_real_robot_novo/admittance/scaler_LSTMmodel_real_part.joblib"
    #     - model for real novo part positon based (/force_dataset/csv_real_robot_position_novo): "/model_real_robot_novo/position/LSTMmodel_real_part.h5"; "/scaler_real_robot_novo/position/scaler_LSTMmodel_real_part.joblib"
    #     - model for simulated data deployed on real robot (force_dataset/csv_simulation/csv_snap_position_interpolated):  model: "model_sim/position/model/LSTMmodel_sim_data.h"             scaler: "model_sim/position/scaler/scaler_LSTM_sim_data.joblib"
    # """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras_tuner.tuners import RandomSearch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import ast  # Library to safely evaluate literal expressions from strings
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"

# List to store dataframes from each CSV file
dataframes = []

# Loop through CSV files in the directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_directory, filename)
        # Read the CSV file into a list of lists
        with open(file_path, 'r') as file:
            lines = file.readlines()
            length_series = len(lines)
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
# print(full_df)
# print(full_df.iloc[1])

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

y = []  # Initialize the target variable

for index, row in full_df.iterrows():
    if (index + 1) % (length_series - 1) == 0:
        y.append(row['Label'])

# Assuming force_df is your feature matrix and labels_df is your target vector
# Concatenate Force_3 column with other relevant features if necessary
X = np.array(force_df['Force_3'].tolist())

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X = X.reshape((len(X) // (length_series - 1), (length_series - 1)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Save the scaler
joblib.dump(scaler, '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/scaler/scaler_LSTM_sim_data.joblib')

# Convert to nparray
X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train_scaled.shape)
print(X_train_scaled)
print(y_train.shape)
print(y_train)



# # LSTM Model===========
# model = Sequential()
# model.add(LSTM(20, input_shape=((length_series - 1), 1)))
# model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (Success or Failure)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_split=0.2)

# # Save the trained model
# model.save('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_real_robot_novo/position/LSTMmodel_real_part.h5')
# ============

# LSTM Model Grid search approach ==============
# Define the model-building function
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=2, max_value=50, step=1), 
                   input_shape=((length_series - 1), 1) ))
    #                ,return_sequences=True))
    # model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))
    # model.add(LSTM(units=hp.Int('units_2', min_value=1, max_value=30, step=1)))
    # model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the HyperModel
hypermodel = build_model

# Define the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=50,
    directory='../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/training2',
    project_name='lstm_hyperparameter_tuning_position')

# Perform hyperparameter tuning
tuner.search(X_train_scaled, y_train, epochs=200, validation_split=0.15)

# Get the best hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=4)[0]

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
best_model.fit(X_train_scaled, y_train, epochs=300, validation_split=0.15)

# Save the best model
best_model.save('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/model/LSTMmodel_sim_data.h5')
# ===================



# Predictions
y_pred_prob = best_model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation
print("LSTM Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot results
# Create a directory to store the plots if it doesn't exist
plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/results'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
    
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
plt.show()

# Plot learning curve and save it
# print(model.history.history())
# plt.figure()
# plt.plot(model.history.history['accuracy'], label='Training Accuracy')
# plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend()
# plt.savefig(os.path.join(plot_directory, 'learning_curve.png'))
# # plt.show()