# Library Imports
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
import json
import os

# Repository Imports
from link_prediction.metrics import evaluate_heuristic
from typesafety import EdgeData, get_weights_filepath, PredType, ModelType, HeuristicType

def compute_sim(items1: set, items2: set):
    """
    Computes the Jaccard similarity between two sets of items.
    
    Args:
        items1 (set): First set of items.
        items2 (set): Second set of items.
    
    Returns:
        float: Jaccard similarity between the two sets.
    """
    union = len(items1.union(items2))
    if union:
        return len(items1.intersection(items2)) / union
    else:
        return 0

def make_correlation_matrix(user_item_dict:dict, num_users:int):
    """ 
    Creates a correlation matrix where each entry (i, j) is the Jaccard similarity
    between user i and user j. This function is multi-threaded.
    
    Args:
        user_item_dict (dict): A dictionary where keys are user IDs and values are sets of item IDs.
        num_users (int): Total number of users.
    
    Returns:
        np.ndarray: Correlation matrix of shape [num_users, num_users].
    """
    correlation_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        if i % 1000 == 999:
            print(f"Processed {i+1} users.")
        for j in range(i+1,num_users):
            similarity = compute_sim(user_item_dict[i], user_item_dict[j])
            correlation_matrix[i][j] = similarity
            correlation_matrix[j][i] = similarity
        # Doing this to avoid counting oneself
        correlation_matrix[i][i] = 0
    return correlation_matrix

def run(edge_data:EdgeData):
    # Initialize User-Item Dict
    user_item_dict = {}

    num_users = edge_data.num_users
    num_items = edge_data.num_items

    pos_train_edges = edge_data.pos_train_edges
    pos_test_edges = edge_data.pos_test_edges

    neg_train_edges = edge_data.neg_train_edges
    neg_test_edges = edge_data.neg_test_edges

    train_edges = torch.cat([pos_train_edges, neg_train_edges], dim=1)
    test_edges = torch.cat([pos_test_edges, neg_test_edges], dim=1)

    test_labels = torch.cat([
        torch.ones(pos_test_edges.size(1), dtype=torch.float32),  # Positive edges
        torch.zeros(neg_test_edges.size(1), dtype=torch.float32)  # Negative edges
    ])

    user_item_dict_filepath = get_weights_filepath(pred_type=PredType.LP, model_type=ModelType.HEURISTIC, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio, heuristic_type=HeuristicType.UID)
    corr_matrix_filepath = get_weights_filepath(pred_type=PredType.LP, model_type=ModelType.HEURISTIC, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio, heuristic_type=HeuristicType.CM)

    if os.path.exists(user_item_dict_filepath):
        with open(user_item_dict_filepath, 'r') as f:
            user_item_dict = json.load(f)
            print('Loaded User-Item Dictionary.')
            # Required because json.dump converts keys to strings
            user_item_dict = {int(k): v for k, v in user_item_dict.items()}            

    else:
        # Create a set of items for each user
        for i in range(num_users):
            user_item_dict[i] = []

        # print("Train set ratings distribution:", train_df['rating'].value_counts(normalize=True))

        # print("Test set ratings distribution:", test_df['rating'].value_counts(normalize=True))

        print('Creating User-Item Dictionary...')

        # Iterates over all rows and creates user item dict by adding items to each user
        for user, item in zip(pos_train_edges[0].tolist(), pos_train_edges[1].tolist()):
            user_item_dict[user].append(item)

        print('Finished Creating User-Item Dictionary...')

        with open(user_item_dict_filepath, 'w') as f:
            json.dump(user_item_dict, f, indent=4)
            print('Saved User-Item Dictionary.')

    # Required because json.dump only accepts lists not sets
    for i in range(num_users):
        user_item_dict[i] = set(user_item_dict[i])


    if os.path.exists(corr_matrix_filepath):
        correlation_matrix = np.load(corr_matrix_filepath)
        print('Loaded Correlation Matrix.')

    else:
        print('Making Correlation Matrix...')

        correlation_matrix = make_correlation_matrix(user_item_dict, num_users)        

        print('Finished Making Correlation Matrix...')

        np.save(corr_matrix_filepath, correlation_matrix)
        print('Saved Correlation Matrix.')


    print('Starting predictions...')

    # Iterate over rows, idxs are non-linear wrt to original ids to use enumerate to count

    split_metrics = evaluate_heuristic(correlation_matrix, user_item_dict, train_edges, test_edges, test_labels, num_users, num_items, k=10)
    for key in split_metrics.keys():
        print(f"{key}: {split_metrics[key]}")

    return split_metrics