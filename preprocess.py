import pandas as pd
import json
import os
import torch
import random

def get_df(file_path:str):
    """
    Binary edges for positive or negative reviews. Greater than 3 -> 1, Less than or equal to 3 -> 0

    Args:
        file_path: File Path to dataset (must be Amazon)

    Returns:
        Returns a df containing user - item interactions, an encoding dictionary to encode user ids to indexes, 
        and an encoding dictionary to encode item ids to indexes

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Load dataset and parse to df
    reviews = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    with open(file_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                reviews.append((r["user_id"], r["asin"], r["rating"]))
            except KeyError as e:
                raise KeyError(f"Missing expected key in JSON data: {e}")

    df = pd.DataFrame(reviews, columns=["user", "item", "rating"])

    print(f"{df['user'].nunique()} users in dataset.")
    print(f"{df['item'].nunique()} items in dataset.")

    return df


def get_edge_indexes_with_ratings_and_neg_edges(file_path:str='amazon/Gift_Cards.jsonl', train_ratio:float=0.8):
    """
    Binary edges for positive or negative reviews. Greater than 3 -> 1, Less than or equal to 3 -> 0
    This specifically does not add an offset for user, item indices so both start from 0

    Args:
        file_path: File Path to dataset (must be Amazon)

    Returns:
        Returns a df containing user - item interactions, an encoding dictionary to encode user ids to indexes, 
        and an encoding dictionary to encode item ids to indexes
    """

    df = get_df(file_path)

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    # User IDs are from 0 to num_users - 1
    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    # Item IDs are from num_users to num_users num_items - 1
    enc_items = {item_id: idx + num_users for idx, item_id in enumerate(df["item"].unique())}

    df["item"] = [enc_items[item_id] for item_id in df["item"]]

    user_ids = torch.tensor(df['user'].values, dtype=torch.int64)
    item_ids = torch.tensor(df['item'].values, dtype=torch.int64)

    edge_indexes = torch.stack([user_ids, item_ids], dim=0)

    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    positive_edges = set(zip(edge_indexes[0].tolist(), edge_indexes[1].tolist()))

    num_negative_edges = 2 * edge_indexes.size(1)

    negative_edges = []

    # This should guarantee MF and GNN observe the same training samples during prediction
    random.seed(13)
    while len(negative_edges) < num_negative_edges:
        user = random.randint(0, num_users - 1)
        item = random.randint(num_users, num_users + num_items - 1)
        if (user, item) not in positive_edges:
            negative_edges.append((user, item))
    
    neg_edge_indexes = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    num_pos_edges = edge_indexes.size(1)
    num_neg_edges = neg_edge_indexes.size(1)

    # Shuffle indices while keeping edge_index and ratings in sync
    torch.manual_seed(17)
    pos_perm = torch.randperm(num_pos_edges)
    torch.manual_seed(31)
    neg_perm = torch.randperm(num_neg_edges)

    num_pos_train = int(train_ratio * num_pos_edges)
    num_neg_train = int(train_ratio * num_neg_edges)

    pos_train_edges = edge_indexes[:, pos_perm[:num_pos_train]]  # Select first 80% edges
    pos_test_edges = edge_indexes[:, pos_perm[num_pos_train:]]   # Select remaining 20%

    neg_train_edges = neg_edge_indexes[:, neg_perm[:num_neg_train]]  # Select first 80% edges
    neg_test_edges = neg_edge_indexes[:, neg_perm[num_neg_train:]]   # Select remaining 20%

    train_ratings = ratings[pos_perm[:num_pos_train]]
    test_ratings = ratings[pos_perm[num_pos_train:]]

    # No subsampling, all of dataset
    subsampling_percent = 1

    return [pos_train_edges, neg_train_edges, pos_test_edges, neg_test_edges, train_ratings, test_ratings, num_users, num_items, subsampling_percent]


def get_subsampled_edge_indexes_with_ratings_and_neg_edges(file_path:str, train_ratio:float, subsampling_percent:float):
    """
    Binary edges for positive or negative reviews. Greater than 3 -> 1, Less than or equal to 3 -> 0
    This specifically does not add an offset for user, item indices so both start from 0

    Args:
        file_path: File Path to dataset (must be Amazon)

    Returns:
        Returns a df containing user - item interactions, an encoding dictionary to encode user ids to indexes, 
        and an encoding dictionary to encode item ids to indexes
    """

    df = get_df(file_path)

    if subsampling_percent != 1:
        enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
        df["user"] = [enc_user[user_id] for user_id in df["user"]]

        user_ids = torch.unique(torch.tensor(df['user'].values, dtype=torch.long))
        num_users = len(user_ids)

        # Save seed for reproducability
        seed = 42
        torch.manual_seed(seed)
        perm = torch.randperm(num_users)

        # Only take 5% as the heuristic is pretty expensive
        user_ids = user_ids[perm[:int(subsampling_percent * num_users)]]
        df = df[df['user'].isin(user_ids.tolist())] 

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    # User IDs are from 0 to num_users - 1
    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    # Item IDs are from num_users to num_users num_items - 1
    enc_items = {item_id: idx + num_users for idx, item_id in enumerate(df["item"].unique())}

    df["item"] = [enc_items[item_id] for item_id in df["item"]]

    user_ids = torch.tensor(df['user'].values, dtype=torch.int64)
    item_ids = torch.tensor(df['item'].values, dtype=torch.int64)

    edge_indexes = torch.stack([user_ids, item_ids], dim=0)

    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    positive_edges = set(zip(edge_indexes[0].tolist(), edge_indexes[1].tolist()))

    num_negative_edges = 2 * edge_indexes.size(1)

    negative_edges = []

    # This should guarantee MF and GNN observe the same training samples during prediction
    random.seed(13)
    while len(negative_edges) < num_negative_edges:
        user = random.randint(0, num_users - 1)
        item = random.randint(num_users, num_users + num_items - 1)
        if (user, item) not in positive_edges:
            negative_edges.append((user, item))
    
    neg_edge_indexes = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    num_pos_edges = edge_indexes.size(1)
    num_neg_edges = neg_edge_indexes.size(1)

    # Shuffle indices while keeping edge_index and ratings in sync
    torch.manual_seed(17)
    pos_perm = torch.randperm(num_pos_edges)
    torch.manual_seed(31)
    neg_perm = torch.randperm(num_neg_edges)

    num_pos_train = int(train_ratio * num_pos_edges)
    num_neg_train = int(train_ratio * num_neg_edges)

    pos_train_edges = edge_indexes[:, pos_perm[:num_pos_train]]  # Select first 80% edges
    pos_test_edges = edge_indexes[:, pos_perm[num_pos_train:]]   # Select remaining 20%

    neg_train_edges = neg_edge_indexes[:, neg_perm[:num_neg_train]]  # Select first 80% edges
    neg_test_edges = neg_edge_indexes[:, neg_perm[num_neg_train:]]   # Select remaining 20%

    train_ratings = ratings[pos_perm[:num_pos_train]]
    test_ratings = ratings[pos_perm[num_pos_train:]]

    return [pos_train_edges, neg_train_edges, pos_test_edges, neg_test_edges, train_ratings, test_ratings, num_users, num_items, subsampling_percent, train_ratio]