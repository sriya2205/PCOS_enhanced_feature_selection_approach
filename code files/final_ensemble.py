import pandas as pd

# Function to perform ensemble feature selection from input CSVs with intersection
def ensemble_feature_selection(input_csv_files, target_column, output_csv='final_ensemble_features_intersection.csv'):
    final_relevant_features_intersection = None

    # Process each input CSV file
    for i, input_csv in enumerate(input_csv_files):
        print(f"\nProcessing {input_csv}...")

        # Load the CSV file
        df = pd.read_csv(input_csv)

        # Ensure the target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {input_csv}")

        # Get the feature columns (ignoring the target column)
        relevant_columns = set(col for col in df.columns if col != target_column)

        # Initialize intersection with the first dataset
        if i == 0:
            final_relevant_features_intersection = relevant_columns
        else:
            # Perform intersection of relevant features
            final_relevant_features_intersection.intersection_update(relevant_columns)

        print(f"Relevant features from {input_csv}: {len(relevant_columns)}")
        print(relevant_columns)

    # Display intersection results
    print("\nFinal relevant features (Intersection) across all datasets:")
    print(final_relevant_features_intersection)

    # Count the total number of relevant features in the intersection
    total_relevant_features = len(final_relevant_features_intersection)
    print(f"Total relevant features (Intersection): {total_relevant_features}")

    # Load the first CSV to get the full dataset for filtering
    full_df = pd.read_csv(input_csv_files[0])

    # Ensure the relevant columns are in the final dataframe (for intersection), including the target column
    final_columns_intersection = list(final_relevant_features_intersection) + [target_column]

    # Filter the dataframe to keep only relevant columns (Intersection)
    filtered_df_intersection = full_df[final_columns_intersection]

    # Save the intersection features into a CSV file
    filtered_df_intersection.to_csv(output_csv, index=False)
    print(f"Filtered data saved (Intersection) to '{output_csv}'")

# Example usage
input_csv_files = ['top_k_chi_square_columns1.csv', 'pearson_features1.csv', 'relief_features1.csv', 'mrmr_features1.csv']
target_column = 'PCOS (Y/N)'  # Make sure the target column matches the cleaned column names
ensemble_feature_selection(input_csv_files, target_column, output_csv='final_ensemble_features_intersection.csv')
