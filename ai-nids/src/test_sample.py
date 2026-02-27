import pandas as pd

df = pd.read_csv("../data/CICIDS2017.csv", low_memory=False)
df.columns = df.columns.str.strip()

# Remove Label column
df = df.drop("Label", axis=1)

# Take 100 random rows
sample = df.sample(100, random_state=42)

sample.to_csv("../data/test_sample_real.csv", index=False)

print("Test file created successfully.")