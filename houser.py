import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocess import get_subsampled_edge_indexes_with_ratings_and_neg_edges
from typesafety import EdgeData, get_houser_weights_filepath, get_weights_filepath, ModelType, PredType, HouserType
from link_prediction.autorun import autorun as autorun_lp
from edge_classification.autorun import autorun as autorun_ec
from learn_alpha import train as learn_alpha
from link_prediction.models.gnn import GCN as lpGCN, run as run_lp
from edge_classification.models.gnn import GCN as ecGCN, run as run_ec
from eval.models.gnn import GCN as evalGCN, run as run_eval
from helpers import harmonic_mean, normalize, combine_with_alpha



data_path = 'data/amazon/Gift_Cards.jsonl'

print('Preprocessing data...')
args = get_subsampled_edge_indexes_with_ratings_and_neg_edges(file_path=data_path, train_ratio=0.8, subsampling_percent=0.05)
print('Finished data preprocessing.')

edge_data = EdgeData.from_args(args)

print(edge_data)

num_users = edge_data.num_users
num_items = edge_data.num_items

pos_train = edge_data.pos_train_edges
pos_test = edge_data.pos_test_edges
neg_train = edge_data.neg_train_edges
neg_test = edge_data.neg_test_edges

train_edges = torch.cat([pos_train, neg_train], dim=1)
test_edges = torch.cat([pos_test, neg_test], dim=1)



lp_gcn_weights_filepath = get_weights_filepath(pred_type=PredType.LP, model_type=ModelType.GNN, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)
if not os.path.exists(lp_gcn_weights_filepath):
    lp_gcn = run_lp(edge_data=edge_data, with_eval=False)
else: 
    lp_gcn = lpGCN(num_users=num_users, num_items=num_items)
    lp_gcn.load_state_dict(torch.load(lp_gcn_weights_filepath, weights_only=False))

ec_gcn_weights_filepath = get_weights_filepath(pred_type=PredType.EC, model_type=ModelType.GNN, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)
if not os.path.exists(ec_gcn_weights_filepath):
    ec_gcn = run_ec(edge_data=edge_data, with_eval=False)
else:
    ec_gcn = ecGCN(num_users=num_users, num_items=num_items)
    ec_gcn.load_state_dict(torch.load(ec_gcn_weights_filepath, weights_only=False))

lp_gcn.eval()
ec_gcn.eval()

# Pre-processing so 4,5 stars is 1 all else is 0
# This must be done now as we are updating edge_data which must be updated BEFORE training eval
edge_data.update_ratings()

train_ratings = edge_data.train_ratings
test_ratings = edge_data.test_ratings
train_labels = torch.cat([train_ratings, torch.zeros(neg_train.size(1), dtype=torch.float32)])
test_labels = torch.cat([test_ratings, torch.zeros(neg_test.size(1), dtype=torch.float32)])

eval_gcn_weights_filepath = get_weights_filepath(pred_type=PredType.EVAL, model_type=ModelType.GNN, subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)
if not os.path.exists(eval_gcn_weights_filepath):
    eval_gcn = run_eval(edge_data=edge_data, with_eval=False)
else:
    eval_gcn = evalGCN(num_users=num_users, num_items=num_items)
    eval_gcn.load_state_dict(torch.load(eval_gcn_weights_filepath, weights_only=False))
    

model_metrics = {
    HouserType.ALPHA: {
        'recall@k': [],
        'precision@k': [],
        'auc': [],
        'mrr': [],
    },
    HouserType.HARMONIC: {
        'recall@k': [],
        'precision@k': [],
        'auc': [],
        'mrr': [],
    },
    HouserType.EVAL: {
        'recall@k': [],
        'precision@k': [],
        'auc': [],
        'mrr': [],
    },
}

k = 10


alpha = learn_alpha(train_edges=train_edges, test_edges=test_edges, train_labels=train_labels, test_labels=test_labels, lp_gnn=lp_gcn, ec_gnn=ec_gcn)

# weights_filepath = get_houser_weights_filepath(subsampling_percent=edge_data.subsampling_percent, training_split=edge_data.train_ratio)
# model = CombinationModel()

# if not os.path.exists(weights_filepath):
#     train(edge_data=edge_data, train_edges=train_edges, train_labels=train_labels, 
#           test_edges=test_edges, test_labels=test_labels, lp_gnn=lp_gcn, ec_gnn=ec_gcn)
    
# model.load_state_dict(torch.load(weights_filepath, weights_only=False))

with torch.no_grad():
    users = torch.unique(test_edges[0])
    for user in tqdm(users, desc="Evaluating Houser"):

        # Get items the user has interacted with in the training set
        train_items = set(train_edges[1][train_edges[0] == user].tolist())

        # Generate predictions for all unseen items
        all_items = set(range(num_users, num_users + num_items))  # Item indices start from num_users
        unseen_items = sorted(all_items - train_items)

        item_tensor = torch.tensor(unseen_items, dtype=torch.long)
        user_tensor = torch.tensor([user] * int(item_tensor.size(0)), dtype=torch.long)

        edge_index = torch.stack([user_tensor, item_tensor])  # Shape: [2, num_unseen_items]

        # Get relevant items in the test set
        relevant_items = set(test_edges[1][(test_edges[0] == user) & (test_labels == 1)].tolist())


        for model in model_metrics.keys():

            if model == HouserType.ALPHA or model == HouserType.HARMONIC:

                lp_pred_scores = lp_gcn.predict(edge_index)  # Shape: [num_unseen_items]
                lp_pred_scores = normalize(lp_pred_scores)
                ec_pred_scores = ec_gcn.predict(edge_index)

                if model == HouserType.ALPHA:
                    predictions = combine_with_alpha(lp_predictions=lp_pred_scores, ec_predictions=ec_pred_scores, alpha=alpha)

                if model == HouserType.HARMONIC:
                    ec_pred_scores = (ec_pred_scores > 0.5).int()
                    predictions = harmonic_mean(lp_predictions=lp_pred_scores, ec_predictions=ec_pred_scores)
                    
            elif model == HouserType.EVAL:

                predictions = eval_gcn.predict(edge_index)

            predictions = normalize(predictions)

            # Rank items by predicted scores
            _, topk_indices = torch.topk(predictions, k=k)
            topk_items = item_tensor[topk_indices].tolist()


            # Compute Recall@K and Precision@K
            recommended_items = set(topk_items)
            true_positives = len(recommended_items.intersection(relevant_items))

            rank_list = [1 / (idx + 1) for idx, item in enumerate(topk_items) if item in relevant_items]
            if len(rank_list) > 0:
                mrr = max(rank_list)
            else:
                mrr = 0
            model_metrics[model]['mrr'].append(mrr)

            if len(relevant_items) > 0:
                recall = true_positives / len(relevant_items)
                model_metrics[model]['recall@k'].append(recall)

            precision = true_positives / k
            model_metrics[model]['precision@k'].append(precision)

            ground_truth = torch.tensor([1 if item in relevant_items else 0 for item in unseen_items])
            pred = (predictions > 0.5).int()

            if len(torch.unique(ground_truth)) >= 2:  # Check if there are at least two classes
                auc = roc_auc_score(ground_truth, pred)
                model_metrics[model]['auc'].append(auc)

        
for model in model_metrics.keys():
    for metric in model_metrics[model].keys():
        model_metrics[model][metric] = torch.tensor(model_metrics[model][metric], dtype=torch.float64).mean().item()
    recall = model_metrics[model]['recall@k']
    precision = model_metrics[model]['precision@k']
    model_metrics[model]['f1'] = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

for model in model_metrics.keys():
    model_name = f'{model.value.capitalize()} Houser' if model != HouserType.EVAL else 'Evaluation GCN'
    print(f'Metrics for {model_name}')
    for metric in model_metrics[model].keys():
        print(f'\t{metric}: {model_metrics[model][metric]}')

metrics_df = pd.DataFrame({
    'Model': ['Eval GNN', 'Harmonic HOUSER', 'Alpha HOUSER'],
    'Recall@K': [model_metrics[HouserType.EVAL]['recall@k'], model_metrics[HouserType.HARMONIC]['recall@k'], model_metrics[HouserType.ALPHA]['recall@k']],
    'Precision@K': [model_metrics[HouserType.EVAL]['precision@k'], model_metrics[HouserType.HARMONIC]['precision@k'], model_metrics[HouserType.ALPHA]['precision@k']],
    'F1': [model_metrics[HouserType.EVAL]['f1'], model_metrics[HouserType.HARMONIC]['f1'], model_metrics[HouserType.ALPHA]['f1']],
    'AUC': [model_metrics[HouserType.EVAL]['auc'], model_metrics[HouserType.HARMONIC]['auc'], model_metrics[HouserType.ALPHA]['auc']],
    'MRR': [model_metrics[HouserType.EVAL]['mrr'], model_metrics[HouserType.HARMONIC]['mrr'], model_metrics[HouserType.ALPHA]['mrr']],
})

# Melt the DataFrame for easier plotting
metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    

# Plot the metrics
plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_df_melted, palette='viridis')
plt.title('Comparison of Heuristic, MF, and GNN Models')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for better visualization
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

img_filepath = f'metrics/housers_vs_eval.png'
plt.savefig(img_filepath)
print(f"Results are shown in command line and graphs are in {img_filepath} folder")
