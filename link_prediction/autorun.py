import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from link_prediction.models.heuristic import run as run_heuristic
from link_prediction.models.mf import run as run_mf
from link_prediction.models.gnn import run as run_gnn
from typesafety import EdgeData

# THIS MUST BE RUN FROM LINK_PREDICTION DIRECTORY

def autorun(edge_data:EdgeData, gnn_only:bool=False):
    print('\nWe run a total of three models to handle the task of Link Prediction\n')

    if not gnn_only:
        print('Heuristic Model:\n')
        heuristic_metrics = run_heuristic(edge_data)
        print('\n')

        print('Matrix Factorization Model:\n')
        mf_metrics = run_mf(edge_data)
        print('\n')

    print('GCN Model:\n')
    gnn_metrics, gnn_model = run_gnn(edge_data)
    print('\n')

    if not gnn_only:
        metrics_df = pd.DataFrame({
            'Model': ['Heuristic', 'MF', 'GNN'],
            'Recall@K': [heuristic_metrics['recall@k'], mf_metrics['recall@k'], gnn_metrics['recall@k']],
            'Precision@K': [heuristic_metrics['precision@k'], mf_metrics['precision@k'], gnn_metrics['precision@k']],
            'F1': [heuristic_metrics['f1'], mf_metrics['f1'], gnn_metrics['f1']],
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
        plt.savefig('metrics/link_prediction.png')

        print("Results are shown in command line and graphs are in metrics/link_prediction.png folder")

    return gnn_metrics, gnn_model
'''
Pytorch
Pytorch-geometric
Amazon-dataset
Typical Recommender systems
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
'''