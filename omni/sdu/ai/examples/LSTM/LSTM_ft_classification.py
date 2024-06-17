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
from sklearn.preprocessing import StandardScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras_tuner.tuners import RandomSearch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import ast  # Library to safely evaluate literal expressions from strings
import joblib
from keras.optimizers import Adam
from scipy.stats import mode

import matplotlib.pyplot as plt
import seaborn as sns

# # 3 scenarios for snap:
# # a) full simulation data
# data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"
# # b) real data from the robot to do the final validation of the predictions
# data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position_2"
# # c) real data to enlarge simulation data to test if that increases the accuracy
# data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position"

# SAVE_DIRECTORY = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/model'
# PLOT_DIRECTORY = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/results'

# 3 scenarios for pick and place:
# a) full simulation data
data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/sim/admittance_interpolated"
# b) real data from the robot to do the final validation of the predictions
data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance2"
# c) real data to enlarge simulation data to test if that increases the accuracy
data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance"

SAVE_DIRECTORY = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_pick_place/test_training_admittance3/model'
PLOT_DIRECTORY = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_pick_place/test_training_admittance3/results'


def get_dataframe(data_directory):
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
        
    return dataframes, length_series

# Concatenate dataframes into one
dataframes_sim, length_series = get_dataframe(data_directory_sim)
full_df = pd.concat(dataframes_sim, ignore_index=True)

dataframes_val, length_series_val = get_dataframe(data_directory_validation)
full_df_val = pd.concat(dataframes_val, ignore_index=True)

dataframes_real, length_series_real = get_dataframe(data_directory_train_real)
full_df_real = pd.concat(dataframes_real, ignore_index=True)

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_val = pd.DataFrame(full_df_val['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_real = pd.DataFrame(full_df_real['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])

y = []  # Initialize the target variable
y_val = [] # Target variable with validation data taken from the robot
y_real = []

for index, row in full_df.iterrows():
    if (index + 1) % (length_series - 1) == 0:
        y.append(row['Label'])
num_assemblies = len(y)

for index, row in full_df_val.iterrows():
    if (index + 1) % (length_series_val - 1) == 0:
        y_val.append(row['Label'])
num_assemblies_val = len(y_val)

for index, row in full_df_real.iterrows():
    if (index + 1) % (length_series_real - 1) == 0:
        y_real.append(row['Label'])
num_assemblies_real = len(y_real)

# Assuming force_df is your feature matrix and labels_df is your target vector
# Concatenate Force_3 column with other relevant features if necessary
X = np.array(force_df['Force_3'].tolist())
print("X: ", X)
y = np.array(y)

X_val = np.array(force_df_val['Force_3'].tolist())
y_val = np.array(y_val)

X_real = np.array(force_df_real['Force_3'].tolist())
y_real = np.array(y_real)

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X = X.reshape((len(X) // (length_series - 1), (length_series - 1), 1))
y = y.reshape((num_assemblies, 1))

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X_val = X_val.reshape((len(X_val) // (length_series_val - 1), (length_series_val - 1), 1))
y_val = y_val.reshape((num_assemblies_val, 1))

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X_real = X_real.reshape((len(X_real) // (length_series_real - 1), (length_series_real - 1), 1))
print("X_real shape: ", X_real.shape)
y_real = y_real.reshape((num_assemblies_real, 1))
    

# LSTM Random search approach ==============
# Define the model-building function
# def build_model(hp):
#     num_features = 1
#     learning_rate = 0.001 
#     optimizer = Adam(learning_rate=learning_rate)
#     model = Sequential()
#     model.add(LSTM(units=hp.Int('units_1', min_value=2, max_value=70, step=1), 
#                    input_shape=((length_series - 1), num_features) ))
#     #                ,return_sequences=True))
#     # model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)))
#     # model.add(LSTM(units=hp.Int('units_2', min_value=1, max_value=40, step=1)))
#     # model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Define the HyperModel
# hypermodel = build_model

# # Define the tuner
# tuner = RandomSearch(
#     hypermodel,
#     objective='val_loss',
#     max_trials=10,
#     directory='../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/training',
#     project_name='lstm_hyperparameter_tuning_position')

# # Perform hyperparameter tuning
# tuner.search(X_train_scaled, y_train, epochs=75, validation_split=0.1)

# # Get the best hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# # Build the model with the best hyperparameters
# best_model = tuner.hypermodel.build(best_hps)

# # Train the best model
# best_model.fit(X_train_scaled, y_train, epochs=500, validation_split=0.1)

# # Print the best hyperparameters
# print("Best Hyperparameters:")
# print(best_hps.values)

# # Save the best model
# best_model.save('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/model/LSTMmodel_sim_data.keras')
# ===================

# ====== LSTM Non random search approach
def LSTM_model(X_train_scaled, y_train, X_test_scaled, y_test, ratio, window_size, original_shape):
    print("\nTRAINING WITH RATIO OF REAL DATA INTO SIM DATA OF :", ratio)
    # # LSTM Model===========
    # Define the learning rate
    learning_rate = 0.001 
    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential()
    model.add(LSTM(3, input_shape=((length_series-1), 1) ))
    # model.add(LSTM(25, input_shape=(window_size, 1) ))
            # ,return_sequences=True))
    model.add(Dropout(rate=0.2))
    # model.add(BatchNormalization())  # Added BatchNormalization layer
    # model.add(LSTM(1))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))  # binary classification (Success or Failure)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=128, validation_split=0.1)

    # Save the trained model
    save_directory = SAVE_DIRECTORY + str(ratio)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save(os.path.join(save_directory, 'LSTMmodel_sim_data.keras'))

    # Predictions no windowing ====
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Predictions for windowed =====
    # y_pred_windows = model.predict(X_test_scaled)
    # print("y_pred_windows.shape: ", y_pred_windows.shape)
    # y_pred = (y_pred_windows > 0.5).astype(int)
    # print("y_pred.shape: ", y_pred.shape)
    # y_pred_reshaped = np.reshape(y_pred, (78, (4501-window_size+1), 1)) # 78 assemblies, transfrom for windowing, 1
    # mode_values = np.squeeze(np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=y_pred_reshaped))
    # # Ensure that the resulting array has the desired shape (78, 1)
    # y_pred = mode_values.reshape(-1, 1)
    # print("y_pred.shape: ", y_pred.shape)
    

    # Evaluation
    print("LSTM Model with ", ratio*100, " percent of augmentation data :")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot results
    # Create a directory to store the plots if it doesn't exist
    plot_directory = PLOT_DIRECTORY + str(ratio)
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


for i in np.arange(0,9,3):
    np.random.seed(np.random.randint(0, 100))
    ratio = i/10
    
    total = X.shape[0]
    minus_real = X_real.shape[0] - round(ratio*total)
    minus_sim = X.shape[0] - (total - (X_real.shape[0]-minus_real))
    
    sim_ammount = X.shape[0] - minus_sim
    real_ammount = X_real.shape[0] - minus_real

    X_combined = np.concatenate((X[:sim_ammount,:,:], X_real[:real_ammount,:,:]), axis=0)
    y_combined = np.concatenate((y[:sim_ammount,:], y_real[:real_ammount,:]), axis=0)
    print("X[:sim_ammount,:,:], X_real[:real_ammount,:,:]  ", X[:sim_ammount,:,:].shape, X_real[:real_ammount,:,:].shape)
    print("y[:sim_ammount,:], y_real[:real_ammount,:]  ", y[:sim_ammount,:].shape, y_real[:real_ammount,:].shape)
    print("X_combined.shape: ", X_combined.shape)
    print("y_combined.shape: ", y_combined.shape)
    
    # Test using windowing: =========
    # Define window size
    window_size = 100

    # Function to create windows from input data with corresponding labels
    def create_windows_with_labels(data, labels, window_size):
        windows = []
        window_labels = []
        for i in range(len(data)):
            time_series = data[i]
            labels_for_time_series = labels[i]
            for j in range(len(time_series) - window_size + 1):
                windows.append(time_series[j:j + window_size])
                # Use the label corresponding to the last time step of the window
                window_labels.append(labels_for_time_series)
        return np.array(windows), np.array(window_labels)

    # Apply windowing to input data with corresponding labels
    X_windows, y_windows = create_windows_with_labels(X_combined, y_combined, window_size)
    X_windows_val, y_windows_val = create_windows_with_labels(X_val, y_val, window_size)

    # Reshape input data to match LSTM input shape
    X_windows = np.reshape(X_windows, (X_windows.shape[0], window_size, 1))
    # =======================

    X_train, X_test, y_train, y_test = X_combined, X_val, y_combined, y_val
    # X_train, X_test, y_train, y_test = X_windows, X_windows_val, y_windows, y_val

    # # Standardize/Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # # Save the scaler
    # joblib.dump(scaler, '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/scaler2/scaler_LSTM_sim_data.joblib')

    # Convert to nparray
    X_train_scaled = np.array(X_train)
    X_test_scaled = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)    
       
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_test shape:", y_test.shape)
    
    # Train and save the model:
    LSTM_model(X_train_scaled, y_train, X_test_scaled, y_test, ratio, window_size, X_combined.shape)