import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models.heuristic import run as run_heuristic
from models.mf import run as run_mf
from models.gnn import run as run_gnn

# THIS MUST BE RUN FROM LINK_PREDICTION DIRECTORY

print('\nWe will be running three models to handle the task of Rating Prediction\n')
print('First Model: Heuristic Model\n')
heuristic_metrics = run_heuristic()
print('\n')

print('Second Model: Matrix Factorization Model\n')
mf_metrics = run_mf()
print('\n')

print('Third Model: GNN Model with Label Propagation (GCN)\n')
gnn_metrics = run_gnn()
print('\n')


metrics_df = pd.DataFrame({
    'Model': ['Heuristic', 'MF', 'GNN'],
    'Recall@K': [heuristic_metrics['recall@k'], mf_metrics['recall@k'], gnn_metrics['recall@k']],
    'Precision@K': [heuristic_metrics['precision@k'], mf_metrics['precision@k'], gnn_metrics['precision@k']],
    'F1': [heuristic_metrics['f1'], mf_metrics['f1'], gnn_metrics['f1']],
    'MAE': [heuristic_metrics['mae'], mf_metrics['mae'], gnn_metrics['mae']],
    'AUC': [heuristic_metrics['auc'], mf_metrics['auc'], gnn_metrics['auc']],
    'MRR': [heuristic_metrics['mrr'], mf_metrics['mrr'], gnn_metrics['mrr']],
})

# Melt the DataFrame for easier plotting
metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

# Plot the metrics
plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_df_melted, palette='viridis')
plt.title('Comparison of Heuristic, MF, and GNN Models')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for better visualization
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('metrics.png')

print("Results are shown in command line and graphs are in metrics.png folder")
print('Now exiting...')
time.sleep(1)
print('Goodbye!')

'''
Pytorch
Pytorch-geometric
Amazon-dataset
Typical Recommender systems
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
'''