import pandas as pd

INPUT_FILE = "data/expenses.csv"
OUTPUT_FILE = "data/expenses_clean.csv"

# Load dataset
data = pd.read_csv(INPUT_FILE)

print("Original dataset size:", len(data))

# Remove exact duplicate rows
data = data.drop_duplicates()

print("After removing duplicate rows:", len(data))

# Save cleaned dataset
data.to_csv(OUTPUT_FILE, index=False)

print(f"Clean dataset saved to: {OUTPUT_FILE}")