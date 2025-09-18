import pandas as pd

# Load CSV
df = pd.read_csv("questions_df.csv")

# Remove "data/" and ".txt" from corpus_id column
df["corpus_id"] = df["corpus_id"].str.replace("data/", "", regex=False)  # Remove "data/"
df["corpus_id"] = df["corpus_id"].str.replace(".txt", "", regex=False)  # Remove ".txt"

# Save the cleaned CSV
df.to_csv("questions_df.csv", index=False)

print("Fixed corpus_id format. Now it should match corpora_id_paths!")
