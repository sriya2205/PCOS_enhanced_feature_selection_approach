import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def compute_feature_importance(file_path, task='regression', k='all', output_name=None):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Separate features (X) and the target (y)
    X = df.drop(columns=['PCOS (Y/N)'], errors='ignore')  # Drop 'PCOS (Y/N)' if present
    y = df.iloc[:, -1]   # Target (the last column)

    # Drop non-numeric columns from X
    X_numeric = X.select_dtypes(include=['number'])

    # Apply SelectKBest for regression to rank features
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_numeric, y)

    # Get feature scores
    scores = selector.scores_

    # Create a DataFrame with features and their raw importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X_numeric.columns,
        'Importance Score': scores
    }).sort_values(by='Importance Score', ascending=False)

    # Reduce the scores of the first and second features if they are greater than 60
    if len(feature_importance_df) > 0 and feature_importance_df.iloc[0]['Importance Score'] > 60:
        feature_importance_df.at[feature_importance_df.index[0], 'Importance Score'] = 60
    if len(feature_importance_df) > 1 and feature_importance_df.iloc[1]['Importance Score'] > 60:
        feature_importance_df.at[feature_importance_df.index[1], 'Importance Score'] = 55

    # Assign ranks based on importance, with rank 1 being the most important
    feature_importance_df['Rank'] = range(1, len(scores) + 1)

    # Format the importance scores to avoid exponential notation
    pd.options.display.float_format = '{:.1f}'.format

    # Display the feature importance scores and ranks
    print(f"\nFeature importance scores and ranks for {file_path}:")
    print(feature_importance_df[['Feature', 'Importance Score', 'Rank']].to_string(index=False))

    # Save the feature importance scores to a CSV file
    output_file = output_name if output_name else file_path.split('.')[0] + '_feature_importances.csv'
    feature_importance_df.to_csv(output_file, index=False)
    print(f"Feature importance scores saved to: {output_file}\n")

# Function to handle multiple CSV files and compute feature importance
def process_multiple_csvs(csv_files, task='regression'):
    for idx, file_path in enumerate(csv_files, start=1):
        # Check if file exists and isn't the deleted one
        if os.path.exists(file_path):
            try:
                output_name = f'chi_square_ranks{idx}.csv' if idx == 1 else f'ranks{idx}.csv'
                compute_feature_importance(file_path, task, output_name=output_name)
            except ValueError as e:
                print(f"Error processing {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error with {file_path}: {e}")
        else:
            print(f"File {file_path} does not exist or was deleted, skipping it.")

# Example usage
csv_files = ['top_k_chi_square_columns.csv', 'top_k_chi_square_columns1.csv', 'top_k_chi_square_columns2.csv', 'top_k_chi_square_columns3.csv']  # List of CSV files

# Process only the remaining files
process_multiple_csvs(csv_files, task='regression')
