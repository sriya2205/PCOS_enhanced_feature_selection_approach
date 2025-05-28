import pandas as pd

# Function to perform ensemble feature selection from input CSVs
def ensemble_feature_selection(input_csv_files, target_column, output_csv='final_ensemble_features.csv'):
    final_relevant_features = set()

    # Process each input CSV file
    for input_csv in input_csv_files:
        print(f"\nProcessing {input_csv}...")

        # Load the CSV file
        df = pd.read_csv(input_csv)

        # Ensure the target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {input_csv}")

        # Get the feature columns (ignoring the target column)
        relevant_columns = [col for col in df.columns if col != target_column]

        # Add relevant columns to the final set of relevant features
        final_relevant_features.update(relevant_columns)

        print(f"Relevant features from {input_csv}: {len(relevant_columns)}")
        print(relevant_columns)

    # Display final results
    print("\nFinal relevant features across all datasets:")
    print(final_relevant_features)
    print(f"Total relevant features: {len(final_relevant_features)}")

    # Load all CSVs to get the full dataset for filtering
    combined_df = pd.concat([pd.read_csv(file) for file in input_csv_files], axis=1)

    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Ensure the relevant columns are in the final dataframe, including the target column
    final_columns = list(final_relevant_features) + [target_column]

    # Filter the dataframe to keep only relevant columns
    filtered_df = combined_df[final_columns]

    # Save the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"\nFiltered data saved to '{output_csv}'")

# Example usage
input_csv_files = ['top_k_chi_square_columns2.csv', 'pearson_features2.csv', 'relief_features2.csv', 'mrmr_features2.csv']
target_column = 'PCOS (Y/N)'
ensemble_feature_selection(input_csv_files, target_column, output_csv='final_ensemble_features2.csv')
