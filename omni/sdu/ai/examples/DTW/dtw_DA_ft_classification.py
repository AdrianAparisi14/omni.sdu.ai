# This code implements a Fault detection algorithm using dynamic time warping (DTW) and
# discriminant analysis for force-torque and position-orientation assembly data

    # Last update and models generated (18/04/2024): 
    #     - model for real mock novo part (/force_dataset/csv_real_robot_admittance): "DTW/model_real_robot/lda_model2.joblib"
    #     - model for real novo part under admittance control (/force_dataset/csv_real_robot_novo/csv_real_robot_admittance_novo): "/model/admittance/lda_model_real_part_novo_admittance.joblib"
    #     - model for real novo part positon based (/force_dataset/csv_real_robot_novo/csv_real_robot_position_novo): "/model/position/lda_model_real_part_novo_position.joblib"
    #     - model for simulated data deployed on real robot (force_dataset/csv_simulation/csv_snap_position_interpolated):  "DTW/model_sim/position/model/lda_model_sim_data.joblib"
    # """

#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

import pandas as pd
import os
import sys
import numpy as np
from fastdtw import fastdtw
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from omni.sdu.ai.utilities import utils as ut_ai
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# 3 scenarios:
# a) full simulation data
data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"
# b) real data from the robot to do the final validation of the predictions
data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position_2"
# c) real data to enlarge simulation data to test if that increases the accuracy
data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position"

# MODEL_DIR = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_sim/position/model'
# PLOT_DIR = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_sim/position/results'

# # 3 scenarios snap:
# # a) full simulation data
# data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/sim/admittance_interpolated"
# # b) real data from the robot to do the final validation of the predictions
# data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance2"
# # c) real data to enlarge simulation data to test if that increases the accuracy
# data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance"

MODEL_DIR = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_pick_place/admittance/model'
PLOT_DIR = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/test_values_snap_sdu/position/results_case1'


full_df, length_series = ut_ai.get_fulldatafrane_from_directory(data_directory_sim)
full_df_val, length_series_val = ut_ai.get_fulldatafrane_from_directory(data_directory_validation)
full_df_real, length_series_real = ut_ai.get_fulldatafrane_from_directory(data_directory_validation)

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_val = pd.DataFrame(full_df_val['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_real = pd.DataFrame(full_df_real['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Positions'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

# Generate reference curve: uses the mean of a set of successful assemblies
data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position_all_correct"
ref_df, length_series_ref = ut_ai.get_fulldatafrane_from_directory(data_directory)
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
# mean_ref_time_series = np.array(ref_force_df['Force_3'].tolist())
# print(mean_ref_time_series)

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


X = np.array(force_df['Force_3'].tolist())
print("X: ", X)
y = np.array(y)

X_val = np.array(force_df_val['Force_3'].tolist())
y_val = np.array(y_val)

X_real = np.array(force_df_real['Force_3'].tolist())
y_real = np.array(y_real)

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X = X.reshape((len(X) // (length_series - 1), (length_series - 1)))
y = y.reshape((num_assemblies))

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X_val = X_val.reshape((len(X_val) // (length_series_val - 1), (length_series_val - 1)))
y_val = y_val.reshape((num_assemblies_val))

# Reshape X to have dimensions (number of assemblies, (length_series - 1), 1)
X_real = X_real.reshape((len(X_real) // (length_series_real - 1), (length_series_real - 1)))
print("X_real shape: ", X_real.shape)
y_real = y_real.reshape((num_assemblies_real))

    

def DTW(X, mean_ref_time_series):
    # Start DTW
    # Initialize lists to store DTW distances and features for discriminant analysis
    dtw_distances = []
    features = []
            
    print("shape X in function DTW(): ", X.shape)
    for row in X:
        new_curve = np.array(row)
        # print("New row: ", row)
        # print("New curve: ", new_curve)

        # Compute DTW distance
        distance, path = fastdtw(mean_ref_time_series, new_curve)
        dtw_distances.append(distance)
        
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
        
        # Empty the buffer
        buffer_curve = []

        # Extract features for discriminant analysis (you can use other features as well)
        # feature = [new_curve.mean(), new_curve.std(), np.median(new_curve), distance]  # First approach used for features
        # feature = [skew(new_curve), kurtosis(new_curve), np.median(new_curve), distance] # Second approach used for features
        feature = additional_features
        features.append(feature)
    
    # print("Features: ", features)
    
    return features
            
def DA(X_train, X_test, y_train, y_test, ratio):

    print("\nTRAINING WITH RATIO OF REAL DATA INTO SIM DATA OF :", ratio)
    # Train Linear Discriminant Analysis (LDA) model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predictions on the testing set
    y_pred = lda.predict(X_test)

    # Save the trained model
    # Create a directory to store the plots if it doesn't exist
    # model_dir = MODEL_DIR + str(ratio)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # joblib.dump(lda, os.path.join(model_dir,'lda_model_sim_data.joblib'))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:\n", classification_report(y_test, y_pred))


    # # Predict failures using DTW distances and discriminant analysis on the entire dataset
    # for index, (distance, feature) in enumerate(zip(dtw_distances, features)):
    #     prediction = lda.predict([feature])
    #     if prediction == 0:  # Assuming failure is labeled as 1
    #         print(f"Potential Failure Detected for curve at index {index}! DTW Distance: {distance}")
    #     else:
    #         print(f"No Failure Detected for curve at index {index}. DTW Distance: {distance}")
            
            
    # Create a directory to store the plots if it doesn't exist
    plot_directory = PLOT_DIR + "/result" + str(ratio)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
        
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    # Increase font size
    plt.rcParams.update({'font.size': 14}) 
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
    # plt.show()
    
    
for i in np.arange(0,7.5,1.5):
    np.random.seed(np.random.randint(0, 100))
    ratio = i/10
    
    total = X.shape[0]
    minus_real = X_real.shape[0] - round(ratio*total)
    minus_sim = X.shape[0] - (total - (X_real.shape[0]-minus_real))
    
    sim_ammount = X.shape[0] - minus_sim
    real_ammount = X_real.shape[0] - minus_real

    X_combined = np.concatenate((X[:sim_ammount,:], X_real[:real_ammount,:]), axis=0)
    y_combined = np.concatenate((y[:sim_ammount], y_real[:real_ammount]), axis=0)
    print("X[:sim_ammount,:,:], X_real[:real_ammount,:,:]  ", X[:sim_ammount,:].shape, X_real[:real_ammount,:].shape)
    print("y[:sim_ammount,:], y_real[:real_ammount,:]  ", y[:sim_ammount].shape, y_real[:real_ammount].shape)
    print("X_combined.shape: ", X_combined.shape)
    print("y_combined.shape: ", y_combined.shape)
    
    features = DTW(X_combined, mean_ref_time_series)
    features_test = DTW(X_val, mean_ref_time_series)
    # print("Shape X mixed: ", X_combined.shape)
    # Train-test split for discriminant analysis
    # X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = features, features_test, y_combined, y_val
    
    DA(X_train, X_test, y_train, y_test, ratio)