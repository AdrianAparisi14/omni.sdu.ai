# This code implements a Fault detection algorithm using Support Vector Machine and Random Forest

    # Last update and models generated (18/04/2024): 
    #     - model for real mock novo part (/force_dataset/csv_real_robot_admittance): model:"SVM/model_real_robot/svm_model2.joblib", "SVM/model_real_robot/best_rf_model_n_estimators_{best_n_estimators}2.joblib"     scaler:"SVM/scaler/scaler_svm_rf_model2.joblib
    #     - model for real novo part under admittance control (/force_dataset/csv_real_robot_novo/csv_real_robot_admittance_novo): "SVM/real_robot_novo/admittance/SVM/model/svm_model.joblib",  "SVM/real_robot_novo/admittance/RF/model/best_rf_model_n_estimators_{best_n_estimators}.joblib"
    #     - model for real novo part positon based (/force_dataset/csv_real_robot_novo/csv_real_robot_position_novo): "SVM/real_robot_novo/position/SVM/model/svm_model.joblib",  "SVM/real_robot_novo/position/RF/model/best_rf_model_n_estimators_{best_n_estimators}.joblib"
    #     - model for simulated data deployed on real robot (force_dataset/csv_simulation/csv_snap_position_interpolated):  model: "model_sim/position/SVM/model/svm_model_sim_data.joblib", "model_sim/position/RF/model/best_rf_model_n_estimators_{best_n_estimators}_sim_data.joblib"             scaler: "model_sim/position/scaler/scaler_svm_rf_model.joblib"
    # """
#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from tsf import TimeSeriesForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from fastdtw import fastdtw
import joblib
from omni.sdu.ai.utilities import utils as ut_ai
import matplotlib.pyplot as plt
import seaborn as sns

# # 3 scenarios snap:
# # a) full simulation data
# data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"
# # b) real data from the robot to do the final validation of the predictions
# data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position_2"
# # c) real data to enlarge simulation data to test if that increases the accuracy
# data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position"

# SVM_BEST_MODEL = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/SVM/model'
# SVM_PLOT = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/SVM/results'
# RF_BEST_MODEL = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/RF/model'
# RF_PLOT = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/RF/results'

# 3 scenarios snap:
# a) full simulation data
data_directory_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/sim/admittance_interpolated"
# b) real data from the robot to do the final validation of the predictions
data_directory_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance2"
# c) real data to enlarge simulation data to test if that increases the accuracy
data_directory_train_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance"

SVM_BEST_MODEL = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_pick_place/admittance/SVM/model'
SVM_PLOT = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_pick_place/admittance/SVM/results'
RF_BEST_MODEL = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_pick_place/admittance/RF/model'
RF_PLOT = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_pick_place/admittance/RF/results'



full_df, length_series = ut_ai.get_fulldatafrane_from_directory(data_directory_sim)
full_df_val, length_series_val = ut_ai.get_fulldatafrane_from_directory(data_directory_validation)
full_df_real, length_series_real = ut_ai.get_fulldatafrane_from_directory(data_directory_validation)

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_val = pd.DataFrame(full_df_val['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_real = pd.DataFrame(full_df_real['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

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

# Split the data into training and testing sets
# random_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_state)

# Standardize/Normalize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_tra = scaler.transform(X_test)

# # Save the scaler
# joblib.dump(scaler, '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/scaler/scaler_svm_rf_model.joblib')



def SVM_classifier(X_train, X_test, y_train, y_test, ratio):
    # Support Vector Machine (SVM) Classifier =======
    # Define the parameter grid to search the best parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],          # Regularization parameter
        'gamma': ['scale', 'auto'],     # Kernel coefficient
        'kernel': ['rbf'],               # Kernel type
    }

    # Create the SVM model
    random_state = np.random.randint(0, 100)
    svm_model = SVC(random_state=random_state)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Train a new SVM model with the best parameters
    best_svm_model = SVC(**best_params, random_state=random_state)
    best_svm_model.fit(X_train, y_train)

    # Save the best-trained model
    svm_best_model_filename = SVM_BEST_MODEL + str(ratio)
    if not os.path.exists(svm_best_model_filename):
        os.makedirs(svm_best_model_filename)
    joblib.dump(best_svm_model, os.path.join(svm_best_model_filename, 'svm_model_sim_data.joblib'))

    # Predictions
    svm_predictions = best_svm_model.predict(X_test)

    # Evaluation
    print("Support Vector Machine (SVM) Classifier for ratio: ", ratio)
    print("Best Parameters:", best_params)
    print("Random state: ", random_state)
    print("Accuracy:", accuracy_score(y_test, svm_predictions))
    print("Classification Report:\n", classification_report(y_test, svm_predictions))

    # Create a directory to store the plots if it doesn't exist
    plot_directory = SVM_PLOT + str(ratio)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
        
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, svm_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
    plt.show()


def RF_classifier(X_train, X_test, y_train, y_test, ratio):
    # Random Forest Classifier =======

    # Grid search over n_estimators in order to find the best
    param_grid = {'n_estimators': [10, 50, 100, 150, 200]}
    random_state = np.random.randint(0, 100)
    rf_model = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(rf_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameter
    best_n_estimators = grid_search.best_params_['n_estimators']
    print("Best number of estimators:", best_n_estimators)

    # Access the best model from the GridSearchCV object
    best_rf_model = grid_search.best_estimator_

    # Save the best Random Forest model
    rf_best_model_filename = RF_BEST_MODEL + str(ratio)
    if not os.path.exists(rf_best_model_filename):
        os.makedirs(rf_best_model_filename)
    joblib.dump(best_rf_model, os.path.join(rf_best_model_filename, 'best_rf_model_sim_data.joblib'))

    # Predictions
    rf_predictions = best_rf_model.predict(X_test)

    # Evaluation
    print("\nRandom Forest Classifier for ratio: ", ratio)
    print("Random state: ", random_state)
    print("Accuracy:", accuracy_score(y_test, rf_predictions))
    print("Classification Report:\n", classification_report(y_test, rf_predictions))

    # Create a directory to store the plots if it doesn't exist
    plot_directory = RF_PLOT + str(ratio)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
        
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, rf_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
    plt.show()
    
    
# def TSF_classifier(X_train, X_test, y_train, y_test, ratio):
#     # Time Series Forest Classifier TODO: dive more into this approach in the future =======

#     # Grid search over n_estimators in order to find the best
#     param_grid = {'n_estimators': [10, 50, 100, 150, 200]}
#     random_state = np.random.randint(0, 100)
#     rf_model = RandomForestClassifier(random_state=random_state)

#     grid_search = GridSearchCV(rf_model, param_grid, cv=5)
#     grid_search.fit(X_train, y_train)

#     # Print the best hyperparameter
#     best_n_estimators = grid_search.best_params_['n_estimators']
#     print("Best number of estimators:", best_n_estimators)

#     # Access the best model from the GridSearchCV object
#     best_rf_model = grid_search.best_estimator_

#     # Save the best Random Forest model
#     rf_best_model_filename = RF_BEST_MODEL + str(ratio)
#     if not os.path.exists(rf_best_model_filename):
#         os.makedirs(rf_best_model_filename)
#     joblib.dump(best_rf_model, os.path.join(rf_best_model_filename, 'best_rf_model_sim_data.joblib'))

#     # Predictions
#     rf_predictions = best_rf_model.predict(X_test)

#     # Evaluation
#     print("\nRandom Forest Classifier for ratio: ", ratio)
#     print("Random state: ", random_state)
#     print("Accuracy:", accuracy_score(y_test, rf_predictions))
#     print("Classification Report:\n", classification_report(y_test, rf_predictions))

#     # Create a directory to store the plots if it doesn't exist
#     plot_directory = RF_PLOT + str(ratio)
#     if not os.path.exists(plot_directory):
#         os.makedirs(plot_directory)
        
#     # Generate confusion matrix
#     conf_matrix = confusion_matrix(y_test, rf_predictions)

#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.savefig(os.path.join(plot_directory, 'confusion_matrix.png'))
#     plt.show()

    
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
    
    # One way to tackle this could be comparing the entire series understood as a distribution so SVM and RF classify
    # all the points at once
    X_train, X_test, y_train, y_test = X_combined, X_val, y_combined, y_val
    SVM_classifier(X_train, X_test, y_train, y_test, ratio)
    RF_classifier(X_train, X_test, y_train, y_test, ratio)
    
    # A second way could be finding features in the distribution and clasify those features
    # features = DTW(X_combined, mean_ref_time_series)
    # features_test = DTW(X_val, mean_ref_time_series)
    
    # X_train, X_test, y_train, y_test = features, features_test, y_combined, y_val
    
    # SVM_classifier(X_train, X_test, y_train, y_test, ratio)
    # RF_classifier(X_train, X_test, y_train, y_test, ratio)