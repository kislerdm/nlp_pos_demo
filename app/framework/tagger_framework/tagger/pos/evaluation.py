# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score


def model_performance(y_true: List[List[str]],
                      y_pred: List[List[str]]) -> Dict[str, float]:
    """Accuracy calculation function
    
    Args:
      y_true: List of true labels of the tokenized sentese.
      y_pred: List of predicted labels of the tokenized sentese.
      
    Returns:
      Dict of metrics:
      
        {
          "accuracy": float,
          "f1_micro": float,
          "f1_macro": float,
          "f1_weighted": float,
        }
    
    Raises:
      ValueError: Exception occurred when input lists' length don't match.
    """
    if len(y_true) == 0:
        return {}
    
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of input lists don't match.")
    
    def _list_flattener(inpt: List[List[str]]) -> List[str]:
        """Flattener for list of lists into a single list."""
        output = []
        for i in inpt:
            output.extend(i)
        return output

    y_true = _list_flattener(y_true)
    y_pred = _list_flattener(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Numper of tokens don't match between y_true and y_pred.")
    
    try:
        metrics = {
          "accuracy": accuracy_score(y_true, y_pred),
          "f1_micro": f1_score(y_true, y_pred, average='micro'),
          "f1_macro": f1_score(y_true, y_pred, average='macro'),
          "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        }
    except Exception as ex:
        raise Exception(f"Metrics calculation error: {ex}")
    return metrics
