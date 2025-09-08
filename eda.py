import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # for heatmap

# Load preprocessed dataset
df = pd.read_csv("processed_data.csv")

print("Dataset Loaded!")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# 1. Missing values check
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0])

# 2. Class distribution (Attack vs Benign)
if "label" in df.columns:
    print("\nClass Distribution:\n", df['label'].value_counts())

    # Bar graph for class distribution
    plt.figure(figsize=(12,5))
    df['label'].value_counts().plot(kind='bar')
    plt.title("Attack vs Benign Distribution")
    plt.xlabel("Attack Type")
    plt.ylabel("Count")
    plt.show()

# 3. Numerical feature distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df[numeric_cols].hist(figsize=(15,15), bins=40)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# 4. Correlation Heatmap (top 20 features for visibility)
plt.figure(figsize=(15,10))
corr = df[numeric_cols].corr()
sns.heatmap(corr.iloc[:20, :20], cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Top 20 Features)")
plt.show()

# 5. Boxplot of key features (example: Flow_Duration, Total_Fwd_Packets)
# sample_cols = numeric_cols[:5]  # first 5 numeric features for demo
# for col in sample_cols:
#     plt.figure(figsize=(8,4))
#     sns.boxplot(x=df['label'], y=df[col])
#     plt.title(f"Boxplot of {col} vs Label")
#     plt.xticks(rotation=45)
#     plt.show()
