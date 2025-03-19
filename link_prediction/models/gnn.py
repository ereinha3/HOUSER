# Library Imports
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# PyTorch Imports
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
from torch_geometric.loader import DataLoader

# Repository Imports
from link_prediction.metrics import evaluate_gnn
from typesafety import EdgeData, get_weights_filepath, ModelType, PredType


class GCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, hidden_size=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # Create user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Low weight initialization to avoid early convergence to sub-optimal minima
        self.user_embeddings.weight.data.uniform_(-0.01, 0.01) 
        self.item_embeddings.weight.data.uniform_(-0.01, 0.01)

        # GCN layers
        self.conv1 = GCNConv(embedding_dim, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer for edge predictions
        self.edge_fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, edge_index, edge_weight=None):
        # Get embeddings for all users and items
        user_embs = self.user_embeddings(torch.arange(0, self.num_users, device=edge_index.device))
        item_embs = self.item_embeddings(torch.arange(0, self.num_items, device=edge_index.device))

        # Combine embeddings into a single node feature matrix
        x = torch.cat([user_embs, item_embs], dim=0)  # Shape: [num_users + num_items, embedding_dim]

        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Third GCN layer
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # Extract embeddings for the edges in the batch
        user_ids = edge_index[0]
        item_ids = edge_index[1] - self.num_users  # Offset item_ids

        user_embs_batch = user_embs[user_ids]
        item_embs_batch = item_embs[item_ids]
        edge_embs = torch.cat([user_embs_batch, item_embs_batch], dim=1)  # Shape: [num_edges, 2 * embedding_dim]

        # Predict ratings for the batch edges
        pred = self.edge_fc(edge_embs)  # Shape: [num_edges, 1]

        return pred
    
    def predict(self, edge_index):

        # Forward pass through the GCN
        with torch.no_grad():
            pred = self.forward(edge_index)

        return pred.squeeze(-1)
    
    
def run(edge_data:EdgeData, with_eval:bool=True):

    num_users = edge_data.num_users
    num_items = edge_data.num_items

    pos_train_edges = edge_data.pos_train_edges
    pos_test_edges = edge_data.pos_test_edges

    neg_train_edges = edge_data.neg_train_edges
    neg_test_edges = edge_data.neg_test_edges

    train_edges = torch.cat([pos_train_edges, neg_train_edges], dim=1)
    test_edges = torch.cat([pos_test_edges, neg_test_edges], dim=1)

    train_labels = torch.cat([
        torch.ones(pos_train_edges.size(1), dtype=torch.float32),  # Positive edges
        torch.zeros(neg_train_edges.size(1), dtype=torch.float32)  # Negative edges
    ])
    test_labels = torch.cat([
        torch.ones(pos_test_edges.size(1), dtype=torch.float32),  # Positive edges
        torch.zeros(neg_test_edges.size(1), dtype=torch.float32)  # Negative edges
    ])

    # Create graph data structure
    train_graph = Data(
        edge_index=train_edges,
        edge_attr=train_labels,
        num_nodes=num_users + num_items
    )

    test_graph = Data(
        edge_index=test_edges,
        edge_attr=test_labels,
        num_nodes=num_users + num_items
    )

    hidden_size = 64
    embedding_dim = 16
    model = GCN(hidden_size=hidden_size, num_users=num_users, num_items=num_items, 
                embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()  

    num_epochs = 500

    weights_filepath = get_weights_filepath(pred_type=PredType.LP, model_type=ModelType.GNN, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)

    if os.path.exists(weights_filepath):
        model.load_state_dict(torch.load(weights_filepath, weights_only=False))
        print('Loaded existing model weights for GCN.')
        model.eval()
    
    else:
        progress_bar = tqdm(range(num_epochs), desc="Epoch 0")

        temperance = 3
        best_test_loss = float('inf')
        patience = 0

        saved_train_loss, saved_test_loss = 0,0

        for epoch in progress_bar:
            # Update the progress bar description
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            model.train()

            optimizer.zero_grad()

            # Forward pass
            pred = model(train_graph.edge_index)
            
            # Compute loss
            loss = criterion(pred.squeeze(-1), train_graph.edge_attr.squeeze(-1))
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            

            model.eval()
            with torch.no_grad():
                pred = model.predict(test_graph.edge_index)
                test_loss = criterion(pred, test_graph.edge_attr.squeeze(-1)).item()  # Use test_graph for evaluation

            # Early stopping logic
            if test_loss < best_test_loss:
                best_test_loss = test_loss  # Update the best test loss
                torch.save(model.state_dict(), weights_filepath)
                saved_test_loss = test_loss
                saved_train_loss = train_loss
                patience = 0  # Reset patience counter
            else:
                patience += 1  # Increment patience counter

            # Check if patience has exceeded temperance
            if patience >= temperance:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {temperance} epochs).")
                break
        
        if patience < temperance:
            print('Suboptimal model weights saved. Try increasing epoch count for better performance.')
            torch.save(model.state_dict(), weights_filepath)
        
        else:
            print('Saved optimal model weights for GCN.')

        print('(Best) Train Loss of Saved Model:', saved_train_loss)
        print('(Best) Test Loss of Saved Model:', saved_test_loss)
        
        print("Reloading optimal model...")
        model.load_state_dict(torch.load(weights_filepath, weights_only=False))
        print('Loaded optimal model.')


    if with_eval:
        # Evaluate on train and test sets
        split_metrics = evaluate_gnn(model, train_graph, test_graph, num_users, num_items, k=10)
        for key in split_metrics.keys():
            print(f"{key}: {split_metrics[key]}")

        return split_metrics, model
    
    else:
        return model