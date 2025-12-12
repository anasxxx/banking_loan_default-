import pandas as pd

# Load the data
df = pd.read_csv("data/processed/train_features.csv")

# Drop ALL object (string) columns except TARGET and SK_ID_CURR
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Dropping {len(object_cols)} string columns: {object_cols}")
df = df.drop(columns=object_cols)

# Save
df.to_csv("data/processed/train_features.csv", index=False)
print(f"✓ Fixed! Shape: {df.shape}")