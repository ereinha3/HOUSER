# Library Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Repository Imports
from preprocess import get_edge_indexes

class MF(nn.Module):
    # General Matrix Factorization Model

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """
        Args:
            num_users (int): Number of users in the dataset.
            num_items (int): Number of items in the dataset.
            embedding_dim (int): Dimension of embedding vectors for users and items
        """

        # Initialize model parameters   
        super().__init__()
        self.num_users = num_users

        # This creates an embedding for users and an embedding for items (these will be learned)
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Initialize embeddings (initialize over uniform distribution from 0.5 to 1, beneficial as decreases likelihood of 
        # useless / dead features (0s))
        self.user_embeddings.weight.data.uniform_(0.5,1)
        self.item_embeddings.weight.data.uniform_(0.5,1)
        
        # Project the embedding dimensionality vector into 1D space to get prediction between 1 and 0
        self.affine_transform = nn.Linear(in_features=embedding_dim, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # Encode the user and item according to the defined user and item embeddings
        user_embeddings = self.user_embeddings(users)

        items = items - self.num_users
        item_embeddings = self.item_embeddings(items)
        
        # Element wise multiplication (same as dot product)
        out = user_embeddings * item_embeddings

        # Map to 1D to get single prediction
        out = self.affine_transform(out)

        out = self.sigmoid(out)

        return out.squeeze(-1)
    
    def predict(self, users: torch.Tensor, items: torch.Tensor):

        # Forward pass through the GCN
        with torch.no_grad():
            pred = self.forward(users, items)

        return pred
    
class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index, labels):
        """
        Args:
            edge_index (torch.Tensor): Tensor of shape [2, num_edges] containing user-item pairs.
            labels (torch.Tensor): Tensor of shape [num_edges] containing labels (0 or 1).
        """
        self.edge_index = edge_index.t()  # Transpose to [num_edges, 2]
        self.labels = labels

    def __len__(self):
        return self.edge_index.size(0)

    def __getitem__(self, idx):
        user = self.edge_index[idx, 0]  # User ID
        item = self.edge_index[idx, 1]  # Item ID
        label = self.labels[idx]        # Label (0 or 1)
        return user, item, label

def run():
    edge_index, ratings, num_users, num_items = get_edge_indexes()

    num_edges = edge_index.size(1)

    train_ratio = 0.8  # 80% for training, 20% for testing

    num_train = int(num_edges * train_ratio)
    num_test = num_edges - num_train

    # Shuffle indices while keeping edge_index and ratings in sync
    torch.manual_seed(17)
    perm = torch.randperm(num_edges)

    train_edges = edge_index[:, perm[:num_train]]  # Select first 80% edges
    test_edges = edge_index[:, perm[num_train:]]   # Select remaining 20%

    train_labels = ratings[perm[:num_train]]
    test_labels = ratings[perm[num_train:]]

    # Split to train and test
    train_dataset = EdgeDataset(train_edges, train_labels)
    test_dataset = EdgeDataset(test_edges, test_labels)  
        
    batch_size = 16

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    embedding_dim = 16
    model = MF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    
    saved_train_loss = 0
    saved_test_loss = 0
    weights_file_path = 'models/weights/mf_weights.pth'

    if os.path.exists(weights_file_path):
        model.load_state_dict(torch.load(weights_file_path, weights_only=False))
        print('Loaded existing model weights for MF.')
        model.eval()


    else:
        best_test_loss = float('inf')
        num_epochs = 10
        temperance = 3
        patience = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_batch_count = 0
            for users, items, truths in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()
                
                # Compresses output to a single vector instead of a matrix of batch_size vectors of length 1
                outputs = model(users, items)

                # Evaluate loss
                loss = criterion(outputs, truths)
                # Backpropagation and parameter updates
                train_loss += loss.item()
                train_batch_count += 1
                loss.backward()
                optimizer.step()
                
            
            # Turns off dropout and batch normalization
            model.eval()

            test_loss = 0
            test_batch_count = 0
            with torch.no_grad():
                for users, items, truths in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                    pred = model.predict(users, items)
                    test_loss += criterion(pred, truths).item()  # Use test_graph for evaluation
                    test_batch_count += 1

            avg_train_loss = train_loss / train_batch_count
            avg_test_loss = test_loss / test_batch_count
            print('Train Loss:', avg_train_loss)
            print('Test Loss:', avg_test_loss)

            # Early stopping logic
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss  # Update the best test loss
                torch.save(model.state_dict(), weights_file_path)
                print('Saved (current) optimal model weights for MF.')
                saved_train_loss = avg_train_loss
                saved_test_loss = avg_test_loss
                patience = 0  # Reset patience counter
            else:
                patience += 1  # Increment patience counter

            # Check if patience has exceeded temperance
            if patience >= temperance:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {temperance} epochs).")
                break

        if patience < temperance:
            print('Suboptimal model weights saved. Try increasing epoch count for better performance.')
            torch.save(model.state_dict(), weights_file_path)

        print('(Best) Train Loss of Saved Model:', saved_train_loss)
        print('(Best) Test Loss of Saved Model:', saved_test_loss)

        

    if saved_train_loss and saved_test_loss:
        return saved_train_loss, saved_test_loss
    else:
        train_loss = 0
        train_batch_count = 0
        with torch.no_grad():
            for users, items, truths in tqdm(train_dataloader, desc=f"Computing Train Loss"):
                pred = model.predict(users, items)
                train_loss += criterion(pred, truths).item()  # Use test_graph for evaluation
                train_batch_count += 1

        test_loss = 0
        test_batch_count = 0
        with torch.no_grad():
            for users, items, truths in tqdm(test_dataloader, desc=f"Computing Test Loss"):
                pred = model.predict(users, items)
                test_loss += criterion(pred, truths).item()  # Use test_graph for evaluation
                test_batch_count += 1

        avg_train_loss = train_loss / train_batch_count
        avg_test_loss = test_loss / test_batch_count
        print('Train Loss:', avg_train_loss)
        print('Test Loss:', avg_test_loss)

        return avg_train_loss, avg_test_loss