
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("processed_financial_data.csv")

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# Risk score distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Risk Score'], bins=50, kde=True, color='red')
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("risk_score_distribution.png")
