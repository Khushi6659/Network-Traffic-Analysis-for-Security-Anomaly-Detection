import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler

print(" Loading dataset in chunks...")

chunksize = 500000   # adjust depending on your RAM
chunks = pd.read_csv("processed_data.csv", chunksize=chunksize, low_memory=False)

processed_chunks = []
label_mapping = {}
le = LabelEncoder()

# First pass: Encode + collect stats for scaling
scaler = StandardScaler()
first_pass = True

for i, chunk in enumerate(chunks):
    print(f" Processing chunk {i+1} with shape {chunk.shape}")

    # Drop duplicates in chunk
    chunk = chunk.drop_duplicates()
    chunk = chunk.fillna(0)

    # Encode categorical (except label)
    for col in chunk.select_dtypes(include=["object"]).columns:
        if col.lower() != "label":
            chunk[col] = chunk[col].astype(str)
            chunk[col] = le.fit_transform(chunk[col])

    # Encode label
    if "label" in chunk.columns:
        chunk["label"] = chunk["label"].astype(str)
        le_label = LabelEncoder()
        chunk["label"] = le_label.fit_transform(chunk["label"])
        for idx, cls in enumerate(le_label.classes_):
            if cls not in label_mapping:
                label_mapping[cls] = int(idx)

    # Collect numeric stats for scaling
    numeric_cols = chunk.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != "label"]

    if first_pass:
        scaler.partial_fit(chunk[numeric_cols])
        first_pass = False
    else:
        scaler.partial_fit(chunk[numeric_cols])

    processed_chunks.append(chunk)

print(" First pass complete. Scaling stats collected.")

# Second pass: Apply scaling chunk by chunk (memory safe)
final_chunks = []
for i, chunk in enumerate(processed_chunks):
    numeric_cols = chunk.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != "label"]
    chunk[numeric_cols] = scaler.transform(chunk[numeric_cols])
    final_chunks.append(chunk)

# Merge scaled chunks
df = pd.concat(final_chunks, ignore_index=True)
print(" Dataset processed & scaled. Final shape:", df.shape)

# Save processed dataset
df.to_csv("features_data.csv", index=False)
print(" Feature selection completed. Saved as features_data.csv")

# Save label mapping
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

print(" Label mapping saved as label_mapping.json")
