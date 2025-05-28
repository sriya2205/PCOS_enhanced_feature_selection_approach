import csv

def remove_slno_column(input_file, output_file):
    with open(input_file, mode='r') as infile:
        reader = csv.DictReader(infile)
        
        # Get the fieldnames without 'Sl.No'
        fieldnames = [field for field in reader.fieldnames if field.lower() != 'sl.no']
        
        # Display result in the terminal
        print(f"Data without 'Sl.No' column:\n{'-'*50}")
        
        # Write to output CSV file
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row.pop('Sl.No', None)  # Remove 'Sl.No' from the row
                print(row)              # Print the row without 'Sl.No'
                writer.writerow(row)    # Write row without 'Sl.No' to the output file

# Input and output file paths
input_file = 'input.csv'
output_file = 'output.csv'

remove_slno_column(input_file, output_file)
