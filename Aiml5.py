import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_url = 'https://github.com/AmritRajGarg/College_Notes/raw/main/AIML/data.csv'

dataset = pd.read_csv(data_url)

summary_statistics = dataset.describe()
print("Summary Statistics:\n", summary_statistics)

# Visualize missing data with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Create a correlation heatmap for numerical columns
plt.figure(figsize=(10, 6))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Optional: Generate pairplot for numerical variables
numerical_data = dataset.select_dtypes(include='number')
sns.pairplot(numerical_data)
plt.show()

categorical_column = 'gender'  
value_counts = dataset[categorical_column].value_counts()
print(f"Value Counts of {categorical_column}:\n", value_counts)
