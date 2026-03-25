import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("C:\\Users\\Rick\\Downloads\\IAI\\project\\results\\results.csv")

print("\nResults Table:\n")
print(df)

# BAR CHART - F1 SCORE

plt.figure()
plt.bar(df["Model"], df["F1"])
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("Model Comparison (F1 Score)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("f1_comparison.png")
plt.show()

# BAR CHART - ALL METRICS

df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("all_metrics.png")
plt.show()

# CONFUSION MATRIX HEATMAP (SGD BEST MODEL)

# Manually taken from the output
cm = [[3342, 61],
      [15, 158]]

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (SGD Classifier)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()