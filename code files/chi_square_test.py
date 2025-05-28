import pandas as pd
from scipy.stats import chi2_contingency

def chi_square_test(input_csv, target_column, output_csv):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Initialize lists to store relevant and irrelevant columns
    relevant_columns = []
    irrelevant_columns = []

    # Perform Chi-Square test for each column
    for column in df.columns:
        if column != target_column:
            contingency_table = pd.crosstab(df[target_column], df[column])
            chi2, p, dof, ex = chi2_contingency(contingency_table)

            # Assuming a significance level of 0.05
            if p < 0.05:
                relevant_columns.append(column)
            else:
                irrelevant_columns.append(column)

    # Display the results
    print(f"Total columns: {len(df.columns)}")
    print(f"Relevant columns: {len(relevant_columns)}")
    print(f"Irrelevant columns: {len(irrelevant_columns)}")
    print("Relevant columns names:")
    for col in relevant_columns:
        print(f" - {col}")
    print("Irrelevant columns names:")
    for col in irrelevant_columns:
        print(f" - {col}")

    # Create a new DataFrame with relevant columns
    relevant_df = df[[target_column] + relevant_columns]

    # Save the relevant columns to a new CSV file
    relevant_df.to_csv(output_csv, index=False)

# Example usage
input_csv = 'ss.csv'
target_column = 'PCOS (Y/N)'
output_csv = 'chi_square_columns.csv'
chi_square_test(input_csv, target_column,output_csv)
