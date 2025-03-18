# Library Imports
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
import json
import os

# Repository Imports
from preprocess import get_subsampled_edge_indexes
from metrics import evaluate_heuristic

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

def make_correlation_matrix_parallel(user_item_dict:dict, num_users: int, num_threads: int = 4):
    """
    Creates a correlation matrix where each entry (i, j) is the Jaccard similarity
    between user i and user j. This function is multi-threaded.
    
    Args:
        user_item_dict (dict): A dictionary where keys are user IDs and values are sets of item IDs.
        num_users (int): Total number of users.
        num_threads (int): Number of threads to use for parallel computation.
    
    Returns:
        np.ndarray: Correlation matrix of shape [num_users, num_users].
    """
    correlation_matrix = np.zeros((num_users, num_users))
    
    # Function to compute similarity for a single pair (i, j)
    def compute_pair(i, j):
        similarity = compute_sim(user_item_dict[i], user_item_dict[j])
        return i, j, similarity
    
    # Use ThreadPoolExecutor for parallel computation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        # Submit tasks for all pairs (i, j) where i < j
        for i in range(num_users):
            if i % 1000 == 999:
                print(f"Processed {i+1} users.")
            for j in range(i + 1, num_users):
                futures.append(executor.submit(compute_pair, i, j))
        
        # Collect results as they complete
        for future in as_completed(futures):
            i, j, similarity = future.result()
            correlation_matrix[i][j] = similarity
            correlation_matrix[j][i] = similarity
    
    # Set diagonal to 0 (avoiding self-similarity for prediction)
    np.fill_diagonal(correlation_matrix, 0)
    
    return correlation_matrix

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

def run():
    # Initialize User-Item Dict
    user_item_dict = {}

    # Preprocess file to df
    pos_edge_index, neg_edge_index, num_users, num_items = get_subsampled_edge_indexes()

    num_pos_edges = pos_edge_index.size(1)
    num_neg_edges = neg_edge_index.size(1)

    train_ratio = 0.8  # 80% for training, 20% for testing

    num_pos_train = int(num_pos_edges * train_ratio)
    num_neg_train = int(num_neg_edges * train_ratio)
    num_pos_test = num_pos_edges - num_pos_train
    num_neg_test = num_neg_edges - num_neg_train

    # Shuffle indices while keeping edge_index and ratings in sync
    torch.manual_seed(17)
    pos_perm = torch.randperm(num_pos_edges)
    torch.manual_seed(31)
    neg_perm = torch.randperm(num_neg_edges)

    pos_train_edges = pos_edge_index[:, pos_perm[:num_pos_train]]  # Select first 80% edges
    pos_test_edges = pos_edge_index[:, pos_perm[num_pos_train:]]   # Select remaining 20%

    neg_train_edges = neg_edge_index[:, neg_perm[:num_neg_train]]  # Select first 80% edges
    neg_test_edges = neg_edge_index[:, neg_perm[num_neg_train:]]   # Select remaining 20%

    train_edges = torch.cat([pos_train_edges, neg_train_edges], dim=1)
    test_edges = torch.cat([pos_test_edges, neg_test_edges], dim=1)

    train_labels = torch.cat([
        torch.ones(num_pos_train, dtype=torch.float32),  # Positive edges
        torch.zeros(num_neg_train, dtype=torch.float32)  # Negative edges
    ])
    test_labels = torch.cat([
        torch.ones(num_pos_test, dtype=torch.float32),  # Positive edges
        torch.zeros(num_neg_test, dtype=torch.float32)  # Negative edges
    ])

    dict_filepath = 'models/weights/user_item_dict.json'
    corr_matrix_filepath = 'models/weights/corr_matrix.npy'

    if os.path.exists(dict_filepath):
        with open(dict_filepath, 'r') as f:
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

        with open(dict_filepath, 'w') as f:
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

        parallel = False

        print(f"{'Not' if not parallel else ''} Using Parallelized Matrix Creation.")

        # Create the correlation matrix which will be num_user x num_user
        if parallel:
            correlation_matrix = make_correlation_matrix_parallel(user_item_dict, num_users)
        else:
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