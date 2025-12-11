
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("anomaly_predictions.csv")

# Summary statistics
summary = df.groupby('Is Anomaly')[['Amount', 'Risk Score']].mean().reset_index()

# Dashboard mock: bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='Is Anomaly', y='Amount', data=summary, palette='Set2')
plt.title("Average Transaction Amount by Anomaly Status")
plt.xlabel("Anomaly (0=No, 1=Yes)")
plt.ylabel("Average Amount")
plt.tight_layout()
plt.savefig("decision_support_amount_summary.png")
