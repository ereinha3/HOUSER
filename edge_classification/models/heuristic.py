# Library Imports
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
import json
import os

# Repository Imports
from preprocess import get_subsampled_edge_indexes

def compute_sim(items1: set, items2: set):
    """
    Computes the Jaccard similarity between two sets of items.
    
    Args:
        items1 (set): First set of items.
        items2 (set): Second set of items.
    
    Returns:
        float: Jaccard similarity between the two sets.
    """
    items1, items2 = set(items1), set(items2)
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
            similarity = compute_sim(user_item_dict[i][0], user_item_dict[j][0])
            correlation_matrix[i][j] = similarity
            correlation_matrix[j][i] = similarity
        # Doing this to avoid counting oneself
        correlation_matrix[i][i] = 0
    return correlation_matrix

def predict_heuristic(user_item_dict:dict, avg_ratings_dict:dict, correlation_matrix:np.ndarray, user_id:int, item_id:int, k:int=10):
    """
    Predicts whether a user will like an item or not given then top k similar users and whether they liked that item or not
    
    Args:
        user_item_dict (dict): A dictionary where keys are user IDs and values are sets of item IDs.
        correlation_matrix (np.ndarray): Correlation matrix of shape [num_users, num_users].
        user_id (int): Current User ID.
        item_id (int): Current Item ID.
        k (int) (default=10): Number of similar neighbor nodes to observe.
    
    Returns:
        np.ndarray: Correlation matrix of shape [num_users, num_users].
    """
    user_row = correlation_matrix[user_id]

    sorted_indices = np.argsort(user_row)[::-1]  # Sort in descending order
    topk_users = sorted_indices[:k]  # Select top k users
    
    rating_sum = 0
    count = 0
    for id in topk_users:
        if item_id in user_item_dict[id][0]:
            rating_sum += user_item_dict[id][1][user_item_dict[id][0].index(item_id)]
            count += 1
    
    if count:
        return rating_sum / count
    else:
        return avg_ratings_dict[item_id]

def run():
    user_item_dict = {}
    avg_ratings_dict = {}

    # Preprocess file to df
    edge_index, ratings, num_users, num_items = get_subsampled_edge_indexes()

    num_edges = edge_index.size(1)

    train_ratio = 0.8  # 80% for training, 20% for testing

    num_train = int(num_edges * train_ratio)
    num_test = num_edges - num_train

    # Shuffle indices while keeping edge_index and ratings in sync
    torch.manual_seed(17)
    perm = torch.randperm(num_edges)

    train_edges = edge_index[:, perm[:num_train]]  # Select first 80% edges
    test_edges = edge_index[:, perm[num_train:]]   # Select remaining 20%

    train_ratings = ratings[perm[:num_train]]
    test_ratings = ratings[perm[num_train:]]


    user_item_dict_filepath = 'models/weights/user_item_dict.json'
    avg_ratings_dict_filepath = 'models/weights/avg_ratings_dict.json'
    corr_matrix_filepath = 'models/weights/corr_matrix.npy'  

    if os.path.exists(avg_ratings_dict_filepath):
        with open(avg_ratings_dict_filepath, 'r') as f:
            avg_ratings_dict = json.load(f)
            print('Loaded Average Rating Dictionary.')
            # Required because json.dump converts keys to strings
            avg_ratings_dict = {int(k): v for k, v in avg_ratings_dict.items()}           

    else:
        # Create a set of items for each user
        # Come back and make this dict[user_id] a dictionary instead of a 2d array
        for i in range(num_users, num_users + num_items):
            avg_ratings_dict[i] = [0,0]

        print('Creating Average Rating Dictionary...')

        # Iterates over all rows and creates user item dict by adding items to each user
        for _, item, rating in zip(train_edges[0].tolist(), train_edges[1].tolist(), train_ratings.tolist()):
            avg_ratings_dict[item][0] += rating
            avg_ratings_dict[item][1] += 1 
        
        for key in avg_ratings_dict.keys():
            rating_sum = avg_ratings_dict[key][0]
            count = avg_ratings_dict[key][1]
            if count:
                avg_ratings_dict[key] = rating_sum / count
            else:
                avg_ratings_dict[key] = 0.5 # True average rating on [0,1] normalized scale

        print('Finished Average Rating Dictionary...')

        with open(avg_ratings_dict_filepath, 'w') as f:
            json.dump(avg_ratings_dict, f, indent=4)
            print('Saved Average Rating Dictionary.')

    if os.path.exists(user_item_dict_filepath):
        with open(user_item_dict_filepath, 'r') as f:
            user_item_dict = json.load(f)
            print('Loaded User-Item Dictionary.')
            # Required because json.dump converts keys to strings
            user_item_dict = {int(k): v for k, v in user_item_dict.items()}           

    else:
        # Create a set of items for each user
        for i in range(num_users):
            user_item_dict[i] = [[],[]]

        # print("Train set ratings distribution:", train_df['rating'].value_counts(normalize=True))

        # print("Test set ratings distribution:", test_df['rating'].value_counts(normalize=True))

        print('Creating User-Item Dictionary...')

        # Iterates over all rows and creates user item dict by adding items to each user
        for user, item, rating in zip(train_edges[0].tolist(), train_edges[1].tolist(), train_ratings.tolist()):
            user_item_dict[user][0].append(item)
            user_item_dict[user][1].append(rating)

        print('Finished Creating User-Item Dictionary...')

        with open(user_item_dict_filepath, 'w') as f:
            json.dump(user_item_dict, f, indent=4)
            print('Saved User-Item Dictionary.')


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

    train_predictions = []
    train_truths = []
    for user, item, rating in zip(train_edges[0].tolist(), train_edges[1].tolist(), train_ratings.tolist()):
            train_predictions.append(predict_heuristic(user_item_dict=user_item_dict, avg_ratings_dict=avg_ratings_dict, correlation_matrix=correlation_matrix, user_id=user, item_id=item))
            train_truths.append(rating)

    train_predictions = torch.tensor(train_predictions, dtype=torch.float64)
    train_truths = torch.tensor(train_truths, dtype=torch.float64)

    test_predictions = []
    test_truths = []
    for user, item, rating in zip(test_edges[0].tolist(), test_edges[1].tolist(), test_ratings.tolist()):
            test_predictions.append(predict_heuristic(user_item_dict=user_item_dict, avg_ratings_dict=avg_ratings_dict, correlation_matrix=correlation_matrix, user_id=user, item_id=item))
            test_truths.append(rating)

    test_predictions = torch.tensor(test_predictions, dtype=torch.float64)
    test_truths = torch.tensor(test_truths, dtype=torch.float64)

    print('Finished predictions.')

    mse_loss = torch.nn.MSELoss()

    final_train_loss = mse_loss(train_predictions, train_truths).item()
    final_test_loss = mse_loss(test_predictions, test_truths).item()
    print('Final Train Loss:', final_train_loss)
    print('Final Test Loss:', final_test_loss)

    return final_train_loss, final_test_loss

run()