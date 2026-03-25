import pandas as pd

df = pd.read_csv("C:\\Users\\Rick\\Downloads\\IAI\\project\\data\\fake_job_postings.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFraud distribution:\n", df['fraudulent'].value_counts())