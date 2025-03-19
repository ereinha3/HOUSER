import torch.nn as nn
import torch
from tqdm import tqdm
from typesafety import EdgeData, get_houser_weights_filepath, HouserDataset
from link_prediction.models.gnn import GCN as lpGCN
from edge_classification.models.gnn import GCN as ecGCN

class CombinationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, comb_pred):
        x = torch.relu(self.fc1(comb_pred))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze(1)

def train(edge_data:EdgeData, train_edges:torch.tensor, train_labels:torch.tensor, test_edges:torch.tensor, test_labels:torch.tensor, lp_gnn:lpGCN, ec_gnn:ecGCN):
    model = CombinationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print(train_edges.shape)
    print(train_labels.shape)
    print(test_edges.shape)
    print(test_labels.shape)

    temperance = 3
    best_test_loss = float('inf')
    patience = 0
    saved_train_loss, saved_test_loss = 0,0

    weights_filepath = get_houser_weights_filepath(subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)

    num_epochs = 2000
    batch_size = 16
    print('Training Houser...')
    progress_bar = tqdm(range(num_epochs), desc="Epoch 0")

    for epoch in progress_bar:
        # Update the progress bar description
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        lp_predictions = lp_gnn.predict(train_edges)
        ec_predictions = ec_gnn.predict(train_edges)
        combined_prections = torch.cat([lp_predictions.unsqueeze(1), ec_predictions.unsqueeze(1)], dim=1)

        train_data = HouserDataset(combined_prections, train_labels)

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        train_loss = 0
        train_batch_count = 0
        test_loss = 0
        test_batch_count = 0

        for preds, labels in train_dataloader:
            predictions = model(preds)
            loss = criterion(predictions, labels)

            train_batch_count += 1
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            

        model.eval()
        with torch.no_grad():
            lp_predictions = lp_gnn.predict(test_edges)
            ec_predictions = ec_gnn.predict(test_edges)
            combined_prections = torch.cat([lp_predictions.unsqueeze(1), ec_predictions.unsqueeze(1)], dim=1)

            test_data = HouserDataset(combined_prections, test_labels)

            test_dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
            )

            for preds, labels in test_dataloader:
                predictions = model(preds)
                test_loss += criterion(predictions, labels).item()
                test_batch_count += 1

        avg_train_loss = train_loss / train_batch_count
        avg_test_loss = test_loss / test_batch_count

        # Early stopping logic
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss  # Update the best test loss
            torch.save(model.state_dict(), weights_filepath)
            saved_test_loss = avg_test_loss
            saved_train_loss = avg_train_loss
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
        print('Saved optimal model weights for Houser.')

    print('(Best) Train Loss of Saved Model:', saved_train_loss)
    print('(Best) Test Loss of Saved Model:', saved_test_loss)
    
