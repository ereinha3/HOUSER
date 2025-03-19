import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from edge_classification.models.heuristic import run as run_heuristic
from edge_classification.models.mf import run as run_mf
from edge_classification.models.gnn import run as run_gnn
from typesafety import EdgeData

def autorun(edge_data:EdgeData, gnn_only:bool=False):
    print('\nWe will run a total of three models to handle the task of Edge (Rating) Classification\n')

    if not gnn_only:
        print('First Model: Heuristic Model\n')
        heuristic_train_loss, heuristic_test_loss = run_heuristic(edge_data)
        print('\n')

        print('Second Model: Matrix Factorization Model\n')
        mf_train_loss, mf_test_loss = run_mf(edge_data)
        print('\n')

    print('Third Model: GNN Model with Label Propagation (GCN)\n')
    gnn_train_loss, gnn_test_loss, model = run_gnn(edge_data)
    print('\n')

    if not gnn_only:
        # Create a plot
        plt.figure(figsize=(10, 6))

        # Plot training losses
        plt.plot(['Heuristic', 'Matrix Factorization', 'GNN'], 
                [heuristic_train_loss, mf_train_loss, gnn_train_loss], 
                marker='o', label='Training Loss', color='blue')

        # Plot test losses
        plt.plot(['Heuristic', 'Matrix Factorization', 'GNN'], 
                [heuristic_test_loss, mf_test_loss, gnn_test_loss], 
                marker='o', label='Test Loss', color='red')

        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('MSE Loss')
        plt.title('Training and Test Losses for Heuristic, Matrix Factorization, and GNN Models')
        plt.legend()
        plt.grid(True)

        # Save the plot to the 'metrics' folder
        plt.savefig('metrics/edge_classification.png')
        plt.close()

        print("Results are shown in command line and graphs are in metrics/edge_classification.png folder")

    return model
'''
Pytorch
Pytorch-geometric
Amazon-dataset
Typical Recommender systems
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
'''