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
    

def parse_data_binary(file_path:str='amazon/Gift_Cards.jsonl'):
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

    # Normalize ratings to be within [0,1]
    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    # Recommend if rating is 4 or 5 stars
    df["rating"] = (df["rating"] < 0.5).astype(int)

    # User IDs are from 0 to num_users - 1
    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    # Item IDs are from 0 to num_items - 1
    enc_items = {item_id: idx for idx, item_id in enumerate(df["item"].unique())}
    df["item"] = [enc_items[item_id] for item_id in df["item"]]

    return df, enc_user, enc_items

def parse_data_scaled(file_path:str='amazon/Gift_Cards.jsonl'):
    """
    Normalized edges corresponding to positive or negative reviews. Normalization -> (rating - 3) / 2 
    This specifically does add an offset for user, item indices so item indices start from num_users

    Args:
        file_path: File Path to dataset (must be Amazon)

    Returns:
        Returns a df containing user - item interactions, an encoding dictionary to encode user ids to indexes, 
        and an encoding dictionary to encode item ids to indexes
    """

    df = get_df(file_path)

    # Normalize ratings to be within [0,1]
    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    # User IDs are from 0 to num_items - 1
    enc_user = {user_id: idx for idx, user_id in enumerate(df["user"].unique())}
    df["user"] = [enc_user[user_id] for user_id in df["user"]]

    num_users = len(enc_user.keys())

    # Item IDs are from 0 to num_users - 1
    enc_items = {item_id: idx + num_users for idx, item_id in enumerate(df["item"].unique())}
    df["item"] = [enc_items[item_id] for item_id in df["item"]]

    return df, enc_user, enc_items

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

    user_ids = torch.tensor(df['user'].values, dtype=torch.int64)
    item_ids = torch.tensor(df['item'].values, dtype=torch.int64)

    edge_indexes = torch.stack([user_ids, item_ids], dim=0)

    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    return edge_indexes, ratings, num_users, num_items

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

    min_rating, max_rating = df['rating'].min(), df["rating"].max()
    df['rating'] = (df["rating"] - min_rating) / (max_rating - min_rating)

    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    return edge_indexes, ratings, num_users, num_items