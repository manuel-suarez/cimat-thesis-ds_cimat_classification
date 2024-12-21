import torch


def calculate_metrics(outputs, targets):
    """
    Calculate metrics for binary semantic segmentation.

    Args:
        outputs (torch.Tensor): Model predictions (logits), shape: (batch_size, 1, H, W).
        targets (torch.Tensor): Ground truth masks, shape: (batch_size, 1, H, W).

    Returns:
        dict: Dictionary containing accuracy, specificity, sensitivity, dice, and IoU scores
    """
    # Binarize predictions
    threshold = torch.mean(outputs).sum()
    preds = (outputs > threshold).float()

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate TP, TN, FP, FN
    TP = (preds * targets).sum().item()  # True Positives
    TN = ((1 - preds) * (1 - targets)).sum().item()  # True Negative
    FP = (preds * (1 - targets)).sum().item()  # False Positives
    FN = ((1 - preds) * targets).sum().item()  # False Negatives

    # Metrics calculation
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    specificity = TN / (TN + FP + 1e-7)
    sensitivity = TP / (TP + FN + 1e-7)  # Recall
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)

    return {
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "dice": dice,
        "iou": iou,
    }
