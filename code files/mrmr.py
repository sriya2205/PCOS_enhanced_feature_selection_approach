import pandas as pd
import pymrmr

def mrmr_feature_filter(input_csv, target_column, num_features=10, output_csv='mrmr_filtered_output.csv'):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Check if the target column exists in the dataset
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the dataset.")

    # Separate the target column from the features
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Ensure all features are numeric; fill NaNs if needed
    features = features.apply(pd.to_numeric, errors='coerce')
    features.fillna(features.mean(), inplace=True)

    # Combine the target column with features for mRMR input
    df_combined = pd.concat([target, features], axis=1)

    # Perform mRMR feature selection
    try:
        selected_features = pymrmr.mRMR(df_combined, 'MIQ', num_features)
    except Exception as e:
        raise RuntimeError(f"Error during mRMR feature selection: {e}")

    # Ensure the target column is included in the selected features
    if target_column not in selected_features:
        selected_features.insert(0, target_column)

    # Identify irrelevant columns
    irrelevant_columns = [col for col in df.columns if col not in selected_features]

    # Display the results
    print("\n" + "="*50)
    print(f"Total columns in the dataset: {len(df.columns)}")
    print("\nRelevant columns:")
    for col in selected_features:
        print(f"  - {col}")
    print(f"\nNumber of relevant columns: {len(selected_features)}")
    print("\nIrrelevant columns:")
    for col in irrelevant_columns:
        print(f"  - {col}")
    print(f"\nNumber of irrelevant columns: {len(irrelevant_columns)}")
    print("="*50 + "\n")

    # Filter the dataframe to keep only relevant columns
    filtered_df = df[selected_features]

    # Save the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data has been saved to '{output_csv}'")

# Example usage
mrmr_feature_filter('ss.csv', 'PCOS (Y/N)', num_features=15)
