import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import KFold
import numpy as np

def calculate_stability_assessment(file_paths, n_splits=5):
    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")

        # Read the CSV file
        data = pd.read_csv(file_path)

        # Initialize dictionaries to store stability assessment values
        spearman_values = {column: [] for column in data.columns if column == 'Importance Score'}
        kendall_values = {column: [] for column in data.columns if column == 'Importance Score'}

        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Perform 5-fold cross-validation
        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index]

            for column in spearman_values.keys():
                spearman_corr, _ = spearmanr(train_data[column], train_data['Rank'])
                kendall_corr, _ = kendalltau(train_data[column], train_data['Rank'])
                spearman_values[column].append(spearman_corr)
                kendall_values[column].append(kendall_corr)

        # Calculate average stability values and multiply by 100
        avg_spearman_values = {column: np.mean([abs(val) for val in values]) * 100 for column, values in spearman_values.items()}
        avg_kendall_values = {column: np.mean([abs(val) for val in values]) * 100 for column, values in kendall_values.items()}

        # Display the results
        print("Average Stability Assessment Values using Spearman Correlation:")
        for feature, value in avg_spearman_values.items():
            print(f"{feature}: {value}")

        print("Average Stability Assessment Values using Kendall Correlation:")
        for feature, value in avg_kendall_values.items():
            print(f"{feature}: {value}")

# Example usage
file_paths = [
    'feature_importances_file9.csv',
    'feature_importances_file10.csv',
    'feature_importances_file11.csv',
    'feature_importances_file12.csv'
]

calculate_stability_assessment(file_paths)
