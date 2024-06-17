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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import ast  # Library to safely evaluate literal expressions from strings
import joblib
from omni.sdu.ai.utilities import utils as ut_ai
import matplotlib.pyplot as plt
import seaborn as sns

data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated"

full_df, length_series = ut_ai.get_fulldatafrane_from_directory(data_directory)
print(full_df)

# Create separate DataFrames for Force and Position
force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
# position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

y = []  # Initialize the target variable

for index, row in full_df.iterrows():
    if (index + 1) % (length_series - 1) == 0:
        y.append(row['Label'])

# Concatenate Force_3 column with other relevant features if necessary
X = np.array(force_df['Force_3'].tolist())

# Reshape X to have dimensions (number of assemblies, (length_series - 1))
X = X.reshape((len(X) // (length_series - 1), (length_series - 1)))

# Split the data into training and testing sets
random_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=random_state)

# Standardize/Normalize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Save the scaler
# joblib.dump(scaler, '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/scaler/scaler_svm_rf_model.joblib')

# Convert to nparray
X_train_scaled = np.array(X_train)
X_test_scaled = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Support Vector Machine (SVM) Classifier =======
# Define the parameter grid to search the best parameters
param_grid = {
    'C': [0.1, 1, 10, 100],          # Regularization parameter
    'gamma': ['scale', 'auto'],     # Kernel coefficient
    'kernel': ['rbf'],               # Kernel type
}

# Create the SVM model
svm_model = SVC(random_state=random_state)

# Create GridSearchCV instance
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

# Perform grid search on the training data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train a new SVM model with the best parameters
best_svm_model = SVC(**best_params, random_state=random_state)
best_svm_model.fit(X_train_scaled, y_train)

# Save the best-trained model
svm_best_model_filename = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/SVM/model/svm_model_sim_data.joblib'
joblib.dump(best_svm_model, svm_best_model_filename)

# Predictions
svm_predictions = best_svm_model.predict(X_test_scaled)

# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# svm_model.fit(X_train_scaled, y_train)

# # Save SVM model
# svm_model_filename = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_real_robot/svm_model.joblib'
# joblib.dump(svm_model, svm_model_filename)

# # Predictions
# svm_predictions = svm_model.predict(X_test_scaled)

# Evaluation
print("Support Vector Machine (SVM) Classifier:")
print("Random state: ", random_state)
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Create a directory to store the plots if it doesn't exist
plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/SVM/results'
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


# Random Forest Classifier =======
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train_scaled, y_train)

# Grid search over n_estimators in order to find the best
param_grid = {'n_estimators': [10, 50, 100, 150, 200]}
rf_model = RandomForestClassifier(random_state=random_state)

grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameter
best_n_estimators = grid_search.best_params_['n_estimators']
print("Best number of estimators:", best_n_estimators)

# Access the best model from the GridSearchCV object
best_rf_model = grid_search.best_estimator_

# Save the best Random Forest model
filename = f'../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/RF/model/best_rf_model_n_estimators_{best_n_estimators}_sim_data.joblib'
joblib.dump(best_rf_model, filename)


# Save Random Forest model
# rf_model_filename = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_real_robot/rf_model.joblib'
# joblib.dump(rf_model, rf_model_filename)

# Predictions
rf_predictions = best_rf_model.predict(X_test_scaled)

# Evaluation
print("\nRandom Forest Classifier:")
print("Random state: ", random_state)
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Create a directory to store the plots if it doesn't exist
plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ai/omni/sdu/ai/examples/SVM/model_sim/position/RF/results'
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