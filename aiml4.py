import pandas as pd

# Define the URL for the CSV file
csv_url = 'https://github.com/ujjawal0402/AIML/blob/main/csvfile.csv'

# Load the data from the URL
df = pd.read_csv(csv_url)

# Export the dataframe to a new CSV file
df.to_csv('exported_data.csv', index=False)

# Print dataset details
row_count, col_count = df.shape
print(f"Number of Rows: {row_count}")
print(f"Number of Columns: {col_count}")
print("First Five Rows:\n", df.head())
print("Size of Dataset:", df.size)

# Display missing values per column
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Analyze numerical columns
numerical_df = df.select_dtypes(include='number')

# Display statistical information
print("Sum of Numerical Columns:\n", numerical_df.sum())
print("Average of Numerical Columns:\n", numerical_df.mean())
print("Minimum Values in Numerical Columns:\n", numerical_df.min())
print("Maximum Values in Numerical Columns:\n", numerical_df.max())
