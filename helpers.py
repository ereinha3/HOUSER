import torch 

def harmonic_mean(lp_predictions: torch.Tensor, ec_predictions: torch.Tensor) -> torch.Tensor:
    """
    Computes the harmonic mean of two tensors, handling division by zero.

    Args:
        lp_predictions (torch.Tensor): Tensor of link prediction scores.
        ec_predictions (torch.Tensor): Tensor of edge classification scores.

    Returns:
        torch.Tensor: Harmonic mean of the input tensors.
    """
    numerator = 2 * lp_predictions * ec_predictions
    denominator = lp_predictions + ec_predictions

    # Handle division by zero
    result = torch.where(
        denominator != 0,  # Condition
        numerator / denominator,  # True case: compute harmonic mean
        torch.tensor(0.0)  # False case: replace with 0.0
    )

    return result



def normalize(input:torch.tensor):
    max_val = torch.max(input)
    min_val = torch.min(input)
    if max_val == min_val:
        raise ZeroDivisionError('All values in tensor are the same')
    return (input - min_val) / (max_val - min_val)

def combine_with_alpha(lp_predictions:torch.tensor, ec_predictions:torch.tensor, alpha:float=0.5):
    return (1 - alpha) * lp_predictions + (alpha) * ec_predictions
