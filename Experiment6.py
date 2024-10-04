```
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Load the dataset
url = "https://github.com/ujjawal0402/AIML/blob/main/csvfile.csv"
df = pd.read_csv(url)

# Check for missing values
print("Missing Values Count:")
print(df.isnull().sum())

# Handle missing values using SimpleImputer (mean strategy)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Check for outliers using Isolation Forest
iforest = IsolationForest(contamination=0.01)
iforest_pred = iforest.fit_predict(df_imputed)

# Remove outliers
df_cleaned = df_imputed[iforest_pred == 1]

# Scale data using RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df_cleaned)

# Convert scaled data back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Check dataset shape
print("Dataset Shape:", df_scaled.shape)

# Save preprocessed dataset
df_scaled.to_csv("preprocessed_csvfile.csv", index=False)
```
