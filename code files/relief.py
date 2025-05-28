import pandas as pd
from skrebate import ReliefF
from sklearn.preprocessing import LabelEncoder, StandardScaler

def relief_feature_filter(input_csv, target_column, num_features=10, output_csv='relief_output.csv'):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Check if the target column exists in the dataset
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the dataset.")

    # Separate features and target column
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical columns to numerical values if needed
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Encode the target column if it is categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform ReliefF feature selection
    relief = ReliefF(n_features_to_select=num_features)
    relief.fit(X_scaled, y)

    # Get the top features selected by ReliefF
    selected_features = X.columns[relief.top_features_[:num_features]]

    # Identify irrelevant columns
    relevant_columns = selected_features.tolist()
    irrelevant_columns = [col for col in X.columns if col not in relevant_columns]

    # Display the results
    print(f"Total columns: {len(X.columns)}")
    print(f"Relevant columns ({len(relevant_columns)}):")
    for col in relevant_columns:
        print(f"  - {col}")
    print(f"Irrelevant columns ({len(irrelevant_columns)}):")
    for col in irrelevant_columns:
        print(f"  - {col}")

    # Filter the dataframe to keep only relevant columns
    filtered_df = df[[target_column] + relevant_columns]

    # Save the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}")

# Example usage
relief_feature_filter('ss.csv', 'PCOS (Y/N)', num_features=15)
