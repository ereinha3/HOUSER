from typing import Tuple
import torch
from enum import Enum

class EdgeData:
    def __init__(
        self,
        pos_train_edges: torch.Tensor,
        neg_train_edges: torch.Tensor,
        pos_test_edges: torch.Tensor,
        neg_test_edges: torch.Tensor,
        train_ratings: torch.Tensor,
        test_ratings: torch.Tensor,
        num_users: int,
        num_items: int,
        subsampling_percent: float,
        train_ratio: float,
    ):
        self._pos_train_edges = pos_train_edges
        self._neg_train_edges = neg_train_edges
        self._pos_test_edges = pos_test_edges
        self._neg_test_edges = neg_test_edges
        self._train_ratings = train_ratings
        self._test_ratings = test_ratings
        self._num_users = num_users
        self._num_items = num_items
        self._subsampling_percent = subsampling_percent
        self._train_ratio = train_ratio

    @property
    def pos_train_edges(self) -> torch.Tensor:
        """Getter for positive training edges."""
        return self._pos_train_edges

    @property
    def neg_train_edges(self) -> torch.Tensor:
        """Getter for negative training edges."""
        return self._neg_train_edges

    @property
    def pos_test_edges(self) -> torch.Tensor:
        """Getter for positive test edges."""
        return self._pos_test_edges

    @property
    def neg_test_edges(self) -> torch.Tensor:
        """Getter for negative test edges."""
        return self._neg_test_edges

    @property
    def train_ratings(self) -> torch.Tensor:
        """Getter for training ratings."""
        return self._train_ratings

    @property
    def test_ratings(self) -> torch.Tensor:
        """Getter for test ratings."""
        return self._test_ratings

    @property
    def num_users(self) -> int:
        """Getter for number of users."""
        return self._num_users

    @property
    def num_items(self) -> int:
        """Getter for number of items."""
        return self._num_items
    
    @property
    def subsampling_percent(self) -> int:
        """Getter for subsampling percent."""
        return self._subsampling_percent
    
    @property
    def train_ratio(self) -> int:
        """Getter for train ratio."""
        return self._train_ratio

    @classmethod
    def from_args(cls, args: list):
        """
        Creates an EdgeData instance from a list of arguments returned by
        `get_edge_indexes_with_ratings_and_neg_edges`.
        """
        return cls(
            pos_train_edges=args[0],
            neg_train_edges=args[1],
            pos_test_edges=args[2],
            neg_test_edges=args[3],
            train_ratings=args[4],
            test_ratings=args[5],
            num_users=args[6],
            num_items=args[7],
            subsampling_percent=args[8],
            train_ratio=args[9],
        )

    def __repr__(self):
        return (
            f"Post-processed data information:\n"
            f"\tpos_train_edges: {self._pos_train_edges.shape},\n"
            f"\tneg_train_edges: {self._neg_train_edges.shape},\n"
            f"\tpos_test_edges: {self._pos_test_edges.shape},\n"
            f"\tneg_test_edges: {self._neg_test_edges.shape},\n"
            f"\ttrain_ratings: {self._train_ratings.shape},\n"
            f"\ttest_ratings: {self._test_ratings.shape},\n"
            f"\tnum_users: {self._num_users},\n"
            f"\tnum_items: {self._num_items}\n"
            f"\tsubsampling_percent: {self._subsampling_percent * 100}%\n"
            f"\ttrain_ratio: {self._train_ratio * 100}%\n"
        )
    
    def update_ratings(self):
        self._train_ratings = (self._train_ratings > 0.5).int()
        self._test_ratings = (self._test_ratings > 0.5).int()
    
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
        user = self.edge_index[idx, 0]  
        item = self.edge_index[idx, 1]  
        label = self.labels[idx]        
        return user, item, label
    
class HouserDataset(torch.utils.data.Dataset):
    def __init__(self, predictions, labels):
        """
        Args:
            predictions (torch.Tensor): Tensor of shape [num_edges, 2] containing LP and EC predictions.
            labels (torch.Tensor): Tensor of shape [num_edges] containing labels (0 or 1).
        """
        self.predictions = predictions
        self.labels = labels

    def __len__(self):
        return self.predictions.size(0)

    def __getitem__(self, idx):
        preds = self.predictions[idx, :]  # Shape: [2]
        label = self.labels[idx]          # Shape: [1]
        return preds, label
    
class ModelType(Enum):
    GNN = 'gnn'
    MF = 'mf'
    HEURISTIC = 'heur'

class PredType(Enum):
    LP = 'link_prediction'
    EC = 'edge_classification'
    EVAL = 'eval'

class HeuristicType(Enum):
    CM = 'corr_matrix'
    UID = 'user_item_dict'
    ARD = 'avg_rating_dict'
    NONE = ''

class HouserType(Enum):
    ALPHA = 'alpha'
    HARMONIC = 'harmonic'
    MLP = 'mlp'


def get_weights_filepath(pred_type:PredType, model_type:ModelType, subsampling_percent:float, training_split:float, heuristic_type:HeuristicType=HeuristicType.NONE) -> str:
    """
    Returns the path to the weights file for the given model type.

    Args:
        model_type (ModelTypes): The type of model (GNN, MF, or HEURISTIC).

    Returns:
        str: The path to the weights file.
    """
    if model_type != ModelType.HEURISTIC:
        extension = '.pth'
    elif heuristic_type == HeuristicType.CM:
        extension = '.npy'
    else:
        extension = '.json'
    pad = '' if heuristic_type == HeuristicType.NONE else '_'
    return f'{pred_type.value}/models/weights/{model_type.value}{pad}{f"{heuristic_type.value}"}_sub_{int(subsampling_percent*100)}_train_rat_{int(training_split * 100)}{extension}'

def get_houser_weights_filepath(subsampling_percent:float, training_split:float) -> str:
    """
    Returns the path to the weights file for the given model type.

    Args:
        model_type (ModelTypes): The type of model (GNN, MF, or HEURISTIC).

    Returns:
        str: The path to the weights file.
    """
    return f'weights/houser_sub_{int(subsampling_percent*100)}_train_rat_{int(training_split * 100)}.pth'