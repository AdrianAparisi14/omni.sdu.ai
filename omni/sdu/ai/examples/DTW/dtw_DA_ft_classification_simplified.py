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
import ast  # Library to safely evaluate literal expressions from strings
from fastdtw import fastdtw
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from omni.sdu.ai.utilities import utils as ut_ai
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"

full_df, length_series = ut_ai.get_fulldatafrane_from_directory(data_directory)
print(full_df)

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Positions'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

# Generate reference curve: uses the mean of a set of successful assemblies
data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_admittance_all_correct"
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


# Initialize lists to store DTW distances and features for discriminant analysis
dtw_distances = []
features = []

# Loop through the curves to calculate DTW distances and extract features
init_index = 0
buffer_curve = [] # To append rows of the dataframe
buffer_labels = [] # Create a buffer to create a single label for each curve
y = []  # Initialize the target variable

for index, row in full_df.iterrows():
    if (index + 1) % (length_series - 1) == 0:
        y.append(row['Label'])

for index, row in force_df.iterrows():
    buffer_curve.append(row['Force_3'])
    if len(buffer_curve) == (length_series - 1):
        new_curve = np.array(buffer_curve)
        # print("New curve: ", new_curve)

        # Compute DTW distance
        distance, path = fastdtw(mean_ref_time_series, new_curve)
        dtw_distances.append(distance)
        
        # Set of multiple features that can be extracted from a time-series
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
        
        # Empty the buffer
        buffer_curve = []

        # Extract features for discriminant analysis (you can use other features as well)
        # feature = [new_curve.mean(), new_curve.std(), np.median(new_curve), distance]  # First approach used for features
        # feature = [skew(new_curve), kurtosis(new_curve), np.median(new_curve), distance] # Second approach used for features
        feature = additional_features
        features.append(feature)
    
# Train-test split for discriminant analysis
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

# Train Linear Discriminant Analysis (LDA) model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predictions on the testing set
y_pred = lda.predict(X_test)

# Save the trained model
model_filename = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_sim/position/model/lda_model_sim_data.joblib'
joblib.dump(lda, model_filename)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))


# Predict failures using DTW distances and discriminant analysis on the entire dataset
for index, (distance, feature) in enumerate(zip(dtw_distances, features)):
    prediction = lda.predict([feature])
    if prediction == 0:  # Assuming failure is labeled as 1
        print(f"Potential Failure Detected for curve at index {index}! DTW Distance: {distance}")
    else:
        print(f"No Failure Detected for curve at index {index}. DTW Distance: {distance}")
        
        
# Create a directory to store the plots if it doesn't exist
plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/DTW/model_sim/position/results'
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