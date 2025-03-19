import torch.nn as nn
import torch
from tqdm import tqdm
from typesafety import EdgeData, get_houser_weights_filepath, HouserDataset
from helpers import normalize
from link_prediction.models.gnn import GCN as lpGCN
from edge_classification.models.gnn import GCN as ecGCN

class CombinationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as a learnable parameter

    def forward(self, param1, param2):
        # Compute the weighted combination: (1 - alpha) * param1 + alpha * param2
        return (1 - self.alpha) * param1 + self.alpha * param2

def train(train_edges: torch.tensor, train_labels: torch.tensor, test_edges: torch.tensor, test_labels: torch.tensor, lp_gnn: lpGCN, ec_gnn: ecGCN):
    model = CombinationModel()
    optimizer = torch.optim.Adam([model.alpha], lr=1e-3)  # Optimize only alpha
    criterion = nn.BCELoss()

    temperance = 3
    best_test_loss = float('inf')
    patience = 0
    saved_train_loss, saved_test_loss = 0, 0

    num_epochs = 2000
    batch_size = 16
    print('Training Houser...')
    progress_bar = tqdm(range(num_epochs), desc="Epoch 0")

    best_alpha = 0

    for epoch in progress_bar:
        # Update the progress bar description
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # Get predictions from the link predictor and edge classifier
        lp_predictions = lp_gnn.predict(train_edges)
        lp_predictions = normalize(lp_predictions)
        ec_predictions = ec_gnn.predict(train_edges)
        ec_predictions = normalize(ec_predictions)

        # Combine predictions into a dataset
        train_data = HouserDataset(torch.cat([lp_predictions.unsqueeze(1), ec_predictions.unsqueeze(1)], dim=1), train_labels)

        # Create a DataLoader with batch_size = 1
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        train_loss = 0
        train_batch_count = 0
        test_loss = 0
        test_batch_count = 0

        # Training loop
        for preds, labels in train_dataloader:
            # Split predictions into param1 and param2
            param1 = preds[:, 0]  # Link prediction output
            param2 = preds[:, 1]  # Edge classification output

            # Compute the weighted combination
            predictions = model(param1, param2)

            # Compute the loss
            loss = criterion(predictions, labels)

            train_batch_count += 1
            train_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Get predictions from the link predictor and edge classifier for the test set
            lp_predictions = lp_gnn.predict(test_edges)
            lp_predictions = normalize(lp_predictions)
            ec_predictions = ec_gnn.predict(test_edges)
            ec_predictions = normalize(ec_predictions)

            # Combine predictions into a dataset
            test_data = HouserDataset(torch.cat([lp_predictions.unsqueeze(1), ec_predictions.unsqueeze(1)], dim=1), test_labels)

            # Create a DataLoader with batch_size = 1
            test_dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
            )

            for preds, labels in test_dataloader:
                # Split predictions into param1 and param2
                param1 = preds[:, 0]  # Link prediction output
                param2 = preds[:, 1]  # Edge classification output

                # Compute the weighted combination
                predictions = model(param1, param2)

                # Compute the loss
                test_loss += criterion(predictions, labels).item()
                test_batch_count += 1

        # Compute average losses
        avg_train_loss = train_loss / train_batch_count
        avg_test_loss = test_loss / test_batch_count

        # Early stopping logic
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss  # Update the best test loss
            best_alpha = model.alpha.item()
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
        best_alpha = model.alpha.item()
    else:
        print('Saved optimal model weights for Houser.')

    print('(Best) Train Loss of Saved Model:', saved_train_loss)
    print('(Best) Test Loss of Saved Model:', saved_test_loss)
    print('Learned alpha:', best_alpha)

    return best_alpha