import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast  # Library to safely evaluate literal expressions from strings
import numpy as np
import csv
import os
from keras.models import load_model
from fastdtw import fastdtw
import joblib
from scipy.stats import skew, kurtosis
from prettytable import PrettyTable
import json



def get_fulldatafrane_from_directory(data_directory):
    """Generates a dataframe from a folder containing csv files from assemblies

    Args:
        data_directory (_type_): directory to the cvs file
        filename (_type_): csv file name

    Returns:
        full_df: output dataframe with columns: 'Forces', 'Positions' and 'Labels'
        length_series: timesteps of each timeseries
    """
    
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
    
    return full_df, length_series
    
    
def get_dataframe_from_directory(data_directory):
    """Generates a dataframe from a single csv file

    Args:
        data_directory (_type_): directory to the cvs file
        filename (_type_): csv file name

    Returns:
        full_df: output dataframe: force, position and label
    """
    # file_path = os.path.join(data_directory, filename)
    file_path = data_directory
    # Read the CSV file into a list of lists
    with open(file_path, 'r') as file:
        lines = file.readlines()
        length_series = len(lines)
    # Convert each line to a list using ast.literal_eval
    data = []
    for line in lines:  # Exclude the last line
        try:
            # Split the line into forces and positions
            forces_str, positions_str = line.strip().split('\t')
            forces = ast.literal_eval(forces_str)
            positions = ast.literal_eval(positions_str)
            data.append((forces, positions))
        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Error: Unable to process line {line} in file {file_path}: {e}")

    # Extract forces and positions
    forces = [item[0][0] for item in data]
    positions = [item[1][0] for item in data]
    # Extract the label from the last line
    # Create a DataFrame for each file
    full_df = pd.DataFrame({'Forces': forces, 'Positions': positions})
    return full_df, length_series


def verification_assembly(csv_file_path, model):
    
    # str variable that specifies whether to use a model created with synthetic data or real data
    _model = model
    
    full_df, length_series = get_dataframe_from_directory(csv_file_path)
    
    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    # Verification with LSTM =========
    LSTM_prediction = verification_LSTM(force_df, length_series, _model)

    # Verification with DTW =========
    DTW_prediction = verification_DTW(force_df, length_series, _model)
    
    # Verification with SVM =========
    SVM_prediction = verification_SVM(force_df, length_series, _model)
    
    # Verification with Random Forest =========
    RF_prediction = verification_RF(force_df, length_series, _model)
        
    # Print outcome
    print_table(LSTM_prediction, DTW_prediction, SVM_prediction, RF_prediction)
    
    # Store predicted result
    result = DTW_prediction
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(result)
        

def verification_LSTM(force_df, length_series, model):
    """Performs verification using LSTM model

    Args:
        force_df (_type_): dataframe of the forces on the tcp
        length_series (_type_): length of each timeseries of the assembly
        model: whether the system will verify with a model trained with sim data or real data 

    Returns:
        y_pred: prediction [[0]] unsuccessful assembly or [[1]] successful assembly
    """
    #Verification with LSTM=========
    # Process only Z-force
    X = np.array(force_df['Force_3'].tolist())

    # Reshape X to have dimensions (number of assemblies, ((length_series)), 1)
    X = X.reshape((len(X) // (length_series), (length_series), 1))
    print("Shape for LSTM: ", X.shape)

    if model == "real":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/scaler/scaler_LSTMmodel2.joblib')
        X_new_scaled = loaded_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

        # If you didn't save the scaler, create a new instance for new data
        # new_scaler = StandardScaler()
        # X_new_scaled = new_scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)        
    
        # Load the model
        model = load_model('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_real_robot/LSTMmodel.h5')

        # Make predictions on the new data
        y_pred_prob = model.predict(X_new_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int)
        if y_pred[0][0] == 0:
            print("\n\nPotential FAILURE detected according to LSTM prediction!")
        elif y_pred[0][0] == 1:
            print("\n\nAssembly SUCCESSFUL according to LSTM prediction")
            
    if model == "novo":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_real_robot_novo/scaler_real_robot_novo/position/scaler_LSTMmodel_real_part.joblib')
        X_new_scaled = loaded_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

        # If you didn't save the scaler, create a new instance for new data
        # new_scaler = StandardScaler()
        # X_new_scaled = new_scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)        
    
        # Load the model
        model = load_model('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_real_robot_novo/position/LSTMmodel_real_part.h5')

        # Make predictions on the new data
        y_pred_prob = model.predict(X_new_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int)
        if y_pred[0][0] == 0:
            print("\n\nPotential FAILURE detected according to LSTM prediction!")
        elif y_pred[0][0] == 1:
            print("\n\nAssembly SUCCESSFUL according to LSTM prediction")
        
    elif model == "sim":
        # Convert to nparray
        X_new_scaled = np.array(X)        
    
        # Load the model
        model = load_model('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model_sim/position/model0.6/LSTMmodel_sim_data.keras')

        # Make predictions on the new data
        y_pred_prob = model.predict(X_new_scaled)
        print("y_pred_prob LSTM: ", y_pred_prob)
        y_pred = (y_pred_prob > 0.5).astype(int)
        print("y_pred LSTM: ", y_pred)
        if y_pred[0][0] == 0:
            print("\n\nPotential FAILURE detected according to LSTM prediction!")
        elif y_pred[0][0] == 1:
            print("\n\nAssembly SUCCESSFUL according to LSTM prediction")
        
    return y_pred


def DTW(X, mean_ref_time_series):

    print("shape X in function DTW(): ", X.shape)
    
    new_curve = np.array(X)

    # Compute DTW distance
    distance, path = fastdtw(mean_ref_time_series, new_curve)
    
    # Set of multiple features that can be extracted from a time-series
    additional_features = [
        # skew(new_curve),
        # kurtosis(new_curve),
        # np.percentile(new_curve, 25),
        # np.percentile(new_curve, 75),
        # np.percentile(new_curve, 75) - np.percentile(new_curve, 25),
        # np.max(np.abs(np.diff(new_curve))),
        # np.median(new_curve),
        # np.median(np.abs(new_curve - np.median(new_curve))),
        new_curve.mean(),
        # new_curve.std(),
        distance,
    ]

    feature = additional_features
        
    return feature, distance


def verification_DTW(force_df, length_series, model):
    """Performs verification using DTM feature and Discriminant Analysis model

    Args:
        force_df (_type_): dataframe of the forces on the tcp
        length_series (_type_): length of each timeseries of the assembly

    Returns:
        y_pred: prediction 0 unsuccessful assembly or 1 successful assembly
    """
    # Take a set of correct assemblies to use their mean as a reference
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_admittance_all_correct"
    ref_df, length_series_ref = get_fulldatafrane_from_directory(data_directory)
    ref_force_df = pd.DataFrame(ref_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])

    ref_time_series = np.empty((0, length_series_ref - 1))  # Initialize an empty 2D array
    buffer_row_ref = []

    for index, row in ref_force_df.iterrows():
        buffer_row_ref.append(row['Force_3'])
        if len(buffer_row_ref) == (length_series_ref - 1):
            ref_time_series = np.vstack((ref_time_series, buffer_row_ref))
            # Empty the buffer
            buffer_row_ref = []

    # create a mean time series based on a few successful assemblies
    mean_ref_time_series = np.mean(ref_time_series, axis=0)
    
    # Create the array containing the new curve
    buffer_curve = [] # To append rows of the dataframe
    for index, row in force_df.iterrows():
        buffer_curve.append(row['Force_3'])

    new_curve = np.array(buffer_curve)

    if model == "real":
        # Compute DTW distance
        distance, path = fastdtw(mean_ref_time_series, new_curve)

        # Extract features for discriminant analysis
        # feature = [new_curve.mean(), new_curve.std(), np.median(new_curve), distance] 
        feature = [skew(new_curve), kurtosis(new_curve), np.median(new_curve), distance]
    
        # Load the saved LDA model
        model_lda_path = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_real_robot/lda_model2.joblib'
        loaded_lda = joblib.load(model_lda_path)
    
    if model == "novo":
        data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_admittance_all_correct"
        ref_df, length_series_ref = get_fulldatafrane_from_directory(data_directory)
        ref_force_df = pd.DataFrame(ref_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])

        ref_time_series = np.empty((0, length_series_ref - 1))  # Initialize an empty 2D array
        buffer_row_ref = []

        for index, row in ref_force_df.iterrows():
            buffer_row_ref.append(row['Force_3'])
            if len(buffer_row_ref) == (length_series_ref - 1):
                ref_time_series = np.vstack((ref_time_series, buffer_row_ref))
                # Empty the buffer
                buffer_row_ref = []
        
        # Compute DTW distance
        distance, path = fastdtw(ref_time_series, new_curve)

        # Extract features for discriminant analysis
        additional_features = [
            skew(new_curve),
            kurtosis(new_curve),
            np.percentile(new_curve, 25),
            np.percentile(new_curve, 75),
            np.percentile(new_curve, 75) - np.percentile(new_curve, 25),
            np.max(np.abs(np.diff(new_curve))),
            np.median(new_curve),
            np.median(np.abs(new_curve - np.median(new_curve))),
            new_curve.mean(),
            new_curve.std(),
            distance,
        ]
        feature = additional_features
        # feature = [skew(new_curve), kurtosis(new_curve), np.median(new_curve), distance]
    
        # Load the saved LDA model
        model_lda_path = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/real_robot_novo/model/position/lda_model_real_part_novo_position.joblib'
        loaded_lda = joblib.load(model_lda_path)
        
    elif model == "sim":
        feature, distance = DTW(new_curve, mean_ref_time_series)
        # Load the saved LDA model
        model_lda_path = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_sim/position/model0.0/lda_model_sim_data.joblib'
        loaded_lda = joblib.load(model_lda_path)
    
    # Predict
    feature = np.array(feature)
    feature = feature.reshape(1, -1)
    DTW_prediction = loaded_lda.predict(feature)
    if DTW_prediction == 0: 
        print(f"\n\n\nPotential Failure Detected during the assembly! DTW Distance: {distance}\n\n\n")
    else:
        print(f"\n\n\nNo Failure Detected during the assembly. DTW Distance: {distance}\n\n\n")
        
    return DTW_prediction

def verification_SVM(force_df, length_series, model):
    """Performs verification using SVM

    Args:
        force_df (_type_): dataframe of the forces on the tcp
        length_series (_type_): length of each timeseries of the assembly

    Returns:
        y_pred: prediction 0 unsuccessful assembly or 1 successful assembly
    """
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())

    # Reshape X to have dimensions (number of assemblies, (length_series))
    X = X.reshape((len(X) // (length_series), (length_series)))

    if model == "real":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/scaler/scaler_svm_rf_model2.joblib')
        X_new_scaled = loaded_scaler.transform(X)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_real_robot/svm_model2.joblib')

    if model == "novo":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/real_robot_novo/scaler/scaler_svm_rf_model.joblib')
        X_new_scaled = loaded_scaler.transform(X)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/real_robot_novo/position/SVM/model/svm_model.joblib')
    
    elif model == "sim":
        # Convert to nparray
        X_new_scaled = np.array(X)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/SVM/model0.0/svm_model_sim_data.joblib')

    # Make predictions on the new data
    SVM_prediction = model.predict(X_new_scaled)
    if SVM_prediction == 0: 
        print(f"\n\n\nPotential Failure Detected during the assembly according to SVM prediction!\n\n\n")
    else:
        print(f"\n\n\nNo Failure Detected during the assembly according to SVM prediction!\n\n\n")
        
    return SVM_prediction

def verification_RF(force_df, length_series, model):
    """Performs verification using Random Forest

    Args:
        force_df (_type_): dataframe of the forces on the tcp
        length_series (_type_): length of each timeseries of the assembly

    Returns:
        y_pred: prediction 0 unsuccessful assembly or 1 successful assembly
    """
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())

    # Reshape X to have dimensions (number of assemblies, (length_series - 1))
    X = X.reshape((len(X) // (length_series), (length_series)))

    if model == "real":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/scaler/scaler_svm_rf_model2.joblib')
        X_new_scaled = loaded_scaler.transform(X)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_real_robot/best_rf_model_n_estimators_502.joblib')
        
    if model == "novo":
        # Use the saved scaler during the training on new data
        loaded_scaler = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/real_robot_novo/scaler/scaler_svm_rf_model.joblib')
        X_new_scaled = loaded_scaler.transform(X)

        # Convert to nparray
        X_new_scaled = np.array(X_new_scaled)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/real_robot_novo/position/RF/model/best_rf_model_n_estimators_100.joblib')
    
    elif model == "sim":
        # Convert to nparray
        X_new_scaled = np.array(X)
        
        # Load the model
        model = joblib.load('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/RF/model0.6/best_rf_model_n_estimators_sim_data.joblib')

    # Make predictions on the new data
    RF_prediction = model.predict(X_new_scaled)
    if RF_prediction == 0:  # Assuming failure is labeled as 1
        print(f"\n\n\nPotential Failure Detected during the assembly according to Random Forest prediction!\n\n\n")
    else:
        print(f"\n\n\nNo Failure Detected during the assembly according to Random Forest prediction!\n\n\n")
        
    return RF_prediction


def print_table(LSTM_pred, DTW_pred, SVM_prediction, RF_prediction):
    if LSTM_pred == 0:
        LSTM_pred = "failure"
    else:
        LSTM_pred = "success"
        
    if DTW_pred == 0:
        DTW_pred = "failure"
    else:
        DTW_pred = "success"
        
    if SVM_prediction == 0:
        SVM_prediction = "failure"
    else:
        SVM_prediction = "success"
        
    if RF_prediction == 0:
        RF_prediction = "failure"
    else:
        RF_prediction = "success"
        
    # Create a PrettyTable object
    table = PrettyTable()

    # Additional options
    table.field_names = ["Verification Method", "Prediction"]
    table.border = True  # Enable borders
    table.align["Name"] = "c"  # Left-align the "Name" column

    # Add rows of data
    table.add_row(["LSTM", LSTM_pred])
    table.add_row(["DTW", DTW_pred])
    table.add_row(["SVM", SVM_prediction])
    table.add_row(["Random Forest", RF_prediction])

    # Print the table
    print(table)



# ====== I don't even know if I am using these functions but I dont dare to delete them: ========

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


# Function to perform inference using the trained model
def predict_with_model(model, input_tensor, device, threshold=0.5):
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # with torch.no_grad():
    #     output = model(input_tensor)

    # # Apply sigmoid activation
    # sigmoid = torch.nn.Sigmoid()
    # probability = sigmoid(output)

    # # Convert to binary prediction based on the threshold
    # prediction = (probability > threshold).float().squeeze().cpu().numpy()
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    print(outputs)
    
    outputs_cpu = outputs.cpu()
    out_array = outputs_cpu.numpy()
    print(out_array)
    # if out_array[0][0] > 0 and out_array[0][0] <= 5.3:
    #     print("\nAssembly SUCCESFULL")
    # else:
    #     print("\nAssembly FAILED")

    # Convert the continuous output to a binary value (0 or 1) using softmax activation
    _, predictions = torch.max(outputs, 1)
    print("Softmax probabilities: ", predictions)
    if predictions[0] == 1:
        print("\nAssembly SUCCESFULL")
    else:
        print("\nAssembly FAILED")
    
    # Convert the continuous output to a binary value (0 or 1) using sigmoid
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(outputs)
    print("sigmoid probabilities: ", probabilities)

    return predictions

# LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Change to 2 for binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out