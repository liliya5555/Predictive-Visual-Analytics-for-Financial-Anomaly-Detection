
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("anomaly_predictions.csv")

# Create graph from sender and receiver
G = nx.from_pandas_edgelist(df.head(200), 'Sender', 'Receiver', create_using=nx.DiGraph())

# Assign risk score as node attribute (avg from related transactions)
risk_scores = df.groupby('Sender')['Risk Score'].mean().to_dict()
nx.set_node_attributes(G, risk_scores, 'risk_score')

# Draw graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.3)
node_colors = [G.nodes[n].get('risk_score', 0.5) for n in G.nodes]
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=100, cmap=plt.cm.viridis)
plt.title("Sender-Receiver Transaction Network with Risk Score Coloring")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Risk Score")
plt.tight_layout()
plt.savefig("transaction_network_graph.png")
