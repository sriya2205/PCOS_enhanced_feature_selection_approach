import pandas as pd
from scipy.stats import pearsonr

def pearson_correlation_filter(input_csv, target_column, threshold=0.1, output_csv='pearson_output.csv'):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the dataset.")

    # Ensure the target column is numeric
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Initialize lists to store relevant and irrelevant columns
    relevant_columns = []
    irrelevant_columns = []

    # Drop rows with NaN values in the target column
    df = df.dropna(subset=[target_column])

    # Iterate through each column and calculate Pearson correlation with the target column
    for column in df.columns:
        if column != target_column:
            # Ensure the column is numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Drop rows with NaN values in the current column
            df_clean = df.dropna(subset=[column])

            # Calculate Pearson correlation
            if len(df_clean) > 1:  # Ensure there are enough data points
                corr, _ = pearsonr(df_clean[target_column], df_clean[column])
                if abs(corr) >= threshold:
                    relevant_columns.append(column)
                else:
                    irrelevant_columns.append(column)
            else:
                irrelevant_columns.append(column)  # Not enough data for correlation

    # Display the results in a readable format
    total_columns = len(df.columns)
    num_relevant = len(relevant_columns)
    num_irrelevant = len(irrelevant_columns)

    print(f"Total columns: {total_columns}")
    print(f"Relevant columns ({num_relevant}):")
    if relevant_columns:
        for col in relevant_columns:
            print(f" - {col}")
    else:
        print(" No relevant columns found.")

    print(f"Irrelevant columns ({num_irrelevant}):")
    if irrelevant_columns:
        for col in irrelevant_columns:
            print(f" - {col}")
    else:
        print(" No irrelevant columns found.")

    # Filter the dataframe to keep only relevant columns
    filtered_df = df[[target_column] + relevant_columns]

    # Save the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"\nFiltered data saved to '{output_csv}'.")

# Example usage
pearson_correlation_filter('ss.csv','PCOS (Y/N)')
