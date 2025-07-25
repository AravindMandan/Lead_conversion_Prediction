from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
def evaluate_classification_model(y_true, y_pred, y_proba=None):
    """
    Evaluate classification performance.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities (only required for AUC).

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1_score": f1_score(y_true, y_pred, average='binary'),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    # Include AUC only if probability scores are provided
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics
