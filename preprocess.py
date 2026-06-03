import pandas as pd

# 1. Load the dataset
df = pd.read_csv("credit_risk_dataset.csv")

print("=" * 50)
print("ORIGINAL DATAFRAME")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nMissing values per column:")
print(df.isnull().sum())

# 2. Fill missing numerical values with median
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"\nFilled '{col}' missing values with median: {median_val}")

print(f"\nMissing values after filling numericals:")
print(df.isnull().sum())

# 3. One-hot encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print("\n" + "=" * 50)
print("ENCODED DATAFRAME")
print("=" * 50)
print(f"Shape: {df_encoded.shape}")
print(f"\nColumns after encoding:")
print(df_encoded.columns.tolist())

# 4. Export
df_encoded.to_csv("credit_risk_cleaned.csv", index=False)
print("\nExported: credit_risk_cleaned.csv")
