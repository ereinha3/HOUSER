import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

def evaluate_gnn(model, train, test, num_users, num_items, k=10):
    """
    Evaluates the model by computing Recall@K and Precision@K for each user.
    
    Args:
        model (nn.Module): The trained GCN model.
        train (Data): Training graph data.
        test (Data): Testing graph data.
        num_users (int): Number of users.
        num_items (int): Number of items.
        k (int): The value of K for Recall@K and Precision@K.
    
    Returns:
        dict: A dictionary containing Recall@K, Precision@K, and F1.
    """
    model.eval()
    recall_list = []
    precision_list = []
    auc_list = []
    mrr_list = []

    with torch.no_grad():
        users = torch.unique(test.edge_index[0])
        # SUBSAMPLING FOR TESTING
        # num_test_users = users.size(0)
        # torch.manual_seed(93)
        # perm = torch.randperm(num_test_users)
        # users = users[perm[:int(num_test_users * 0.5)]]
        for user in tqdm(users, desc="Evaluating"):

            # Get items the user has interacted with in the training set
            train_items = set(train.edge_index[1][train.edge_index[0] == user].tolist())

            # Generate predictions for all unseen items
            all_items = set(range(num_users, num_users + num_items))  # Item indices start from num_users
            unseen_items = sorted(all_items - train_items)

            item_tensor = torch.tensor(unseen_items, dtype=torch.long)
            user_tensor = torch.tensor([user] * int(item_tensor.size(0)), dtype=torch.long)

            edge_index = torch.stack([user_tensor, item_tensor])  # Shape: [2, num_unseen_items]

            pred_scores = model.predict(edge_index)  # Shape: [num_unseen_items]

            # if user==users[0]:
            #     print(pred_scores)

            # Rank items by predicted scores
            _, topk_indices = torch.topk(pred_scores, k=k)
            topk_items = item_tensor[topk_indices].tolist()

            # Get relevant items in the test set
            relevant_items = set(test.edge_index[1][(test.edge_index[0] == user) & (test.edge_attr == 1)].tolist())

            # Compute Recall@K and Precision@K
            recommended_items = set(topk_items)
            true_positives = len(recommended_items.intersection(relevant_items))

            rank_list = [1 / (idx + 1) for idx, item in enumerate(topk_items) if item in relevant_items]
            if len(rank_list) > 0:
                mrr = max(rank_list)
            else:
                mrr = 0
            mrr_list.append(mrr)

            if len(relevant_items) > 0:
                recall = true_positives / len(relevant_items)
                recall_list.append(recall)

            precision = true_positives / k
            precision_list.append(precision)

            ground_truth = torch.tensor([1 if item in relevant_items else 0 for item in unseen_items])
            pred = (pred_scores > 0.5).int()

            if len(torch.unique(ground_truth)) >= 2:  # Check if there are at least two classes
                auc = roc_auc_score(ground_truth, pred)
                auc_list.append(auc)

            

    # Average metrics across all users
    avg_recall = torch.tensor(recall_list).mean().item() if recall_list else 0
    avg_precision = torch.tensor(precision_list).mean().item() if precision_list else 0
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_auc = torch.tensor(auc_list).mean().item() if auc_list else 0
    avg_mrr = torch.tensor(mrr_list).mean().item() if mrr_list else 0

    return {
        'recall@k': avg_recall,
        'precision@k': avg_precision,
        'f1': avg_f1,
        'auc': avg_auc,
        'mrr': avg_mrr,
    }

def evaluate_mf(model, train_edges, test_edges, test_labels, num_users, num_items, k=10):
    """
    Evaluates the model by computing Recall@K and Precision@K for each user.
    
    Args:
        model (nn.Module): The trained MF model.
        train (Data): Training data.
        test (Data): Testing data.
        num_users (int): Number of users.
        num_items (int): Number of items.
        k (int): The value of K for Recall@K and Precision@K.
    
    Returns:
        dict: A dictionary containing Recall@K, Precision@K, and F1.
    """
    model.eval()
    recall_list = []
    precision_list = []
    auc_list = []
    mrr_list = []

    with torch.no_grad():
        users = torch.unique(test_edges[0])
        # SUBSAMPLING FOR TESTING
        # num_test_users = users.size(0)
        # perm = torch.randperm(num_test_users)
        # users = users[perm[:int(num_test_users * 0.05)]]
        for user in tqdm(users, desc="Evaluating"):
            # Get items the user has interacted with in the training set
            train_items = set(train_edges[1][train_edges[0] == user].tolist())

            # Generate predictions for all unseen items
            all_items = set(range(num_users, num_users + num_items))  # Item indices start from num_users
            unseen_items = sorted(all_items - train_items)

            item_tensor = torch.tensor(unseen_items, dtype=torch.long)
            user_tensor = torch.tensor([user] * int(item_tensor.size(0)), dtype=torch.long)

            pred_scores = model.predict(user_tensor, item_tensor)  # Shape: [num_unseen_items]

            # if user==users[0]:
            #     print(pred_scores)

            # Rank items by predicted scores
            _, topk_indices = torch.topk(pred_scores, k=k)
            topk_items = item_tensor[topk_indices].tolist()

            # Get relevant items in the test set
            relevant_items = set(test_edges[1][(test_edges[0] == user) & (test_labels == 1)].tolist())


            # Compute Recall@K and Precision@K
            recommended_items = set(topk_items)

            true_positives = len(recommended_items.intersection(relevant_items))

            rank_list = [1 / (idx + 1) for idx, item in enumerate(topk_items) if item in relevant_items]
            if len(rank_list) > 0:
                mrr = max(rank_list)
            else:
                mrr = 0
            mrr_list.append(mrr)

            if len(relevant_items) > 0:
                recall = true_positives / len(relevant_items)
                recall_list.append(recall)

            precision = true_positives / k
            precision_list.append(precision)

            ground_truth = torch.tensor([1 if item in relevant_items else 0 for item in unseen_items])
            pred = (pred_scores > 0.5).int()

            if len(torch.unique(ground_truth)) >= 2:  # Check if there are at least two classes
                auc = roc_auc_score(ground_truth, pred)
                auc_list.append(auc)

            

    # Average metrics across all users
    avg_recall = torch.tensor(recall_list).mean().item() if recall_list else 0
    avg_precision = torch.tensor(precision_list).mean().item() if precision_list else 0
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_auc = torch.tensor(auc_list).mean().item() if auc_list else 0
    avg_mrr = torch.tensor(mrr_list).mean().item() if mrr_list else 0

    return {
        'recall@k': avg_recall,
        'precision@k': avg_precision,
        'f1': avg_f1,
        'auc': avg_auc,
        'mrr': avg_mrr
    }


def predict_heuristic(user_item_dict:dict, correlation_matrix:np.ndarray, user_id:int, item_id:int, k:int=10):
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
    
    count = 0
    for id in topk_users:
        if item_id in user_item_dict[id]:
            count += 1
    
    percentage = count / len(topk_users)
    
    return percentage

def evaluate_heuristic(correlation_matrix, user_item_dict, train_edges, test_edges, test_labels, num_users, num_items, k=10):
    """
    Evaluates the model by computing Recall@K and Precision@K for each user.
    
    Args:
        correlation_matrix (np.ndarray): Correlation matrix between users and item in train.
        user_item_dict (dict): Dictionary containing user, item pairs.
        train (Data): Training data.
        test (Data): Testing data.
        num_users (int): Number of users.
        num_items (int): Number of items.
        k (int): The value of K for Recall@K and Precision@K.
    
    Returns:
        dict: A dictionary containing Recall@K, Precision@K, and F1.
    """
    recall_list = []
    precision_list = []
    mae_list = []
    auc_list = []
    mrr_list = []

    with torch.no_grad():
        users = torch.unique(test_edges[0])
        # SUBSAMPLING FOR TESTING
        # num_test_users = users.size(0)
        # perm = torch.randperm(num_test_users)
        # users = users[perm[:int(num_test_users * 0.05)]]
        for user in tqdm(users, desc="Evaluating"):
            # Get items the user has interacted with in the training set
            train_items = set(train_edges[1][train_edges[0] == user].tolist())

            # Generate predictions for all unseen items
            all_items = set(range(num_users, num_users + num_items))  # Item indices start from num_users
            unseen_items = sorted(all_items - train_items)

            item_tensor = torch.tensor(unseen_items, dtype=torch.long)

            predictions = []
            for item in unseen_items:
                predictions.append(predict_heuristic(user_item_dict, correlation_matrix, user, item, k=k))

            predictions = torch.tensor(predictions, dtype=torch.float32)

            # Rank items by predicted scores
            _, topk_indices = torch.topk(predictions, k=k)
            topk_items = item_tensor[topk_indices].tolist()

            # Get relevant items in the test set
            relevant_items = set(test_edges[1][(test_edges[0] == user) & (test_labels == 1)].tolist())

            # Compute Recall@K and Precision@K
            recommended_items = set(topk_items)

            true_positives = len(recommended_items.intersection(relevant_items))

            rank_list = [1 / (idx + 1) for idx, item in enumerate(topk_items) if item in relevant_items]
            if len(rank_list) > 0:
                mrr = max(rank_list)
            else:
                mrr = 0
            mrr_list.append(mrr)

            if len(relevant_items) > 0:
                recall = true_positives / len(relevant_items)
                recall_list.append(recall)

            precision = true_positives / k
            precision_list.append(precision)

            ground_truth = torch.tensor([1 if item in relevant_items else 0 for item in unseen_items])

            if len(torch.unique(ground_truth)) >= 2:  # Check if there are at least two classes
                auc = roc_auc_score(ground_truth, predictions)
                auc_list.append(auc)

            predictions = (predictions > 0.5).int()
            # print(predictions)

            correct = (predictions == ground_truth).float().mean()

            

    # Average metrics across all users
    avg_recall = torch.tensor(recall_list).mean().item() if recall_list else 0
    avg_precision = torch.tensor(precision_list).mean().item() if precision_list else 0
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_auc = torch.tensor(auc_list).mean().item() if auc_list else 0
    avg_mrr = torch.tensor(mrr_list).mean().item() if mrr_list else 0

    return {
        'recall@k': avg_recall,
        'precision@k': avg_precision,
        'f1': avg_f1,
        'auc': avg_auc,
        'mrr': avg_mrr,
    }


