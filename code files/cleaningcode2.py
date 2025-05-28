import pandas as pd

# Function to preprocess and clean specific columns in the data
def preprocess_specific_columns(input_file, output_file, bmi_value, fsh_lh_value, waist_hip_value):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Replace 'BMI' column values
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        df['BMI'] = df['BMI'].fillna(bmi_value)
    else:
        print("BMI column not found.")
    
    # Replace 'FSH/LH' column values
    if 'FSH/LH' in df.columns:
        df['FSH/LH'] = pd.to_numeric(df['FSH/LH'], errors='coerce')
        df['FSH/LH'] = df['FSH/LH'].fillna(fsh_lh_value)
    else:
        print("FSH/LH column not found.")
    
    # Replace 'Waist:Hip' column values
    if 'Waist:Hip' in df.columns:
        df['Waist:Hip'] = pd.to_numeric(df['Waist:Hip'], errors='coerce')
        df['Waist:Hip'] = df['Waist:Hip'].fillna(waist_hip_value)
    else:
        print("Waist:Hip column not found.")
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Specific columns have been processed. Output saved to {output_file}.")

# Example usage
input_file = 'input_data.csv'  # Replace with your input CSV file
output_file = 'cleaned_data.csv'  # Replace with your desired output CSV file
bmi_value = 25.0  # Numeric value for 'BMI'
fsh_lh_value = 1.5  # Numeric value for 'FSH/LH'
waist_hip_value = 0.9  # Numeric value for 'Waist:Hip'

preprocess_specific_columns(input_file, output_file, bmi_value, fsh_lh_value, waist_hip_value)
