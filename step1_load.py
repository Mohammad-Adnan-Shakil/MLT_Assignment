# step1_load.py
import os
import pandas as pd

os.makedirs("figures", exist_ok=True)

def load_data(arff_path="Training Dataset.arff"):
    if os.path.exists(arff_path):
        from scipy.io import arff
        raw, _ = arff.loadarff(arff_path)
        df = pd.DataFrame(raw)
        df = df.apply(lambda c: c.str.decode("utf-8").astype(int))
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=327)  # Phishing Websites
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df

df = load_data()
target = df.columns[-1]

print("Shape:", df.shape)
print("\nTarget column:", target)
print("\nClass distribution:")
counts = df[target].value_counts()
print(pd.DataFrame({
    "count": counts,
    "pct": (counts / len(df) * 100).round(2)
}).rename(index={-1: "phishing (-1)", 1: "legitimate (1)"}))

print("\nUnique values per feature:")
print(df.drop(columns=target).nunique().value_counts().sort_index())

print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

df.to_csv("phishing_clean.csv", index=False)