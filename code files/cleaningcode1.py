import pandas as pd

# Function to preprocess and clean specific columns in the data
def preprocess_specific_columns(input_file, output_file, bmi_value, fsh_lh_value):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Check if the 'BMI' column exists and replace its values
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')  # Convert to numeric, force errors to NaN
        df['BMI'] = df['BMI'].fillna(bmi_value)  # Fill NaN with specified value
    else:
        print("BMI column not found.")
    
    # Check if the 'FSH/LH' column exists and replace its values
    if 'FSH/LH' in df.columns:
        df['FSH/LH'] = pd.to_numeric(df['FSH/LH'], errors='coerce')  # Convert to numeric, force errors to NaN
        df['FSH/LH'] = df['FSH/LH'].fillna(fsh_lh_value)  # Fill NaN with specified value
    else:
        print("FSH/LH column not found.")
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Specific columns have been processed. Output saved to {output_file}.")

# Example usage
input_file = 'input_data.csv'  # Replace with your input CSV file
output_file = 'cleaned_data.csv'  # Replace with your desired output CSV file
bmi_value = 25.0  # Replace with the numeric value you want for 'BMI'
fsh_lh_value = 1.5  # Replace with the numeric value you want for 'FSH/LH'

preprocess_specific_columns(input_file, output_file, bmi_value, fsh_lh_value)
