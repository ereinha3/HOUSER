import json
import pandas as pd
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

    return df
    

def get_edge_indexes(file_path:str='amazon/Gift_Cards.jsonl'):
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

    user_ids = torch.tensor(df['user'].values, dtype=torch.long)
    item_ids = torch.tensor(df['item'].values, dtype=torch.long)

    edge_indexes = torch.stack([user_ids, item_ids], dim=0)

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
    
    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    return edge_indexes, negative_edges, num_users, num_items


def get_subsampled_edge_indexes(file_path:str='amazon/Gift_Cards.jsonl'):
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

    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    user_ids = torch.unique(torch.tensor(df['user'].values, dtype=torch.long))
    num_users = len(user_ids)

    # Save seed for reproducability
    seed = 42
    torch.manual_seed(seed)
    perm = torch.randperm(num_users)

    # Only take 5% as the heuristic is pretty expensive
    ratio = 0.05 
    user_ids = user_ids[perm[:int(ratio * num_users)]]
    df = df[df['user'].isin(user_ids.tolist())] 

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    # User IDs are from 0 to num_users - 1
    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    # Item IDs are from num_users to num_users num_items - 1
    enc_items = {item_id: idx + num_users for idx, item_id in enumerate(df["item"].unique())}

    df["item"] = [enc_items[item_id] for item_id in df["item"]]

    user_ids = torch.tensor(df['user'].values, dtype=torch.long)
    item_ids = torch.tensor(df['item'].values, dtype=torch.long)

    edge_indexes = torch.stack([user_ids, item_ids], dim=0)

    positive_edges = set(zip(edge_indexes[0].tolist(), edge_indexes[1].tolist()))

    num_negative_edges = 2 * edge_indexes.size(1)

    negative_edges = []

    random.seed(seed)
    while len(negative_edges) < num_negative_edges:
        user = random.randint(0, num_users - 1)
        item = random.randint(num_users, num_users + num_items - 1)
        if (user, item) not in positive_edges:
            negative_edges.append((user, item))
    
    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    return edge_indexes, negative_edges, num_users, num_items

