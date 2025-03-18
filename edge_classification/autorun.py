import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.heuristic import run as run_heuristic
from models.mf import run as run_mf
from models.gnn import run as run_gnn

# THIS MUST BE RUN FROM EDGE_CLASSIFICATION DIRECTORY

print('\nWe will be running three models to handle the task of Edge (Rating) Classification\n')
print('First Model: Heuristic Model\n')
heuristic_train_loss, heuristic_test_loss = run_heuristic()
print('\n')

print('Second Model: Matrix Factorization Model\n')
mf_train_loss, mf_test_loss = run_mf()
print('\n')

print('Third Model: GNN Model with Label Propagation (GCN)\n')
gnn_train_loss, gnn_test_loss = run_gnn()
print('\n')

print("Results are shown in command line and graphs are in 'metrics' folder")
print('Now exiting...')
time.sleep(1)
print('Goodbye!')

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
plt.savefig('metrics.png')
plt.close()


'''
Pytorch
Pytorch-geometric
Amazon-dataset
Typical Recommender systems
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
'''