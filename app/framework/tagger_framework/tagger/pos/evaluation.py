# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List
import numpy


def accuracy(y_true: List[List[str]], 
             y_pred: List[List[str]]) -> float:
    """Accuracy calculation function
    
    Args:
      y_true: List of true labels of the tokenized sentese.
      y_pred: List of predicted labels of the tokenized sentese.
      
    Returns:
      Accuracy in the range between 0.0 and 1.0, 
      or None in case of an empty input.
    
    Raises:
      ValueError: Exception occurred when input lists' length don't match.
    """
    if len(y_true) == 0:
        return None
    
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of input lists don't match.")
    
    def _list_flattener(inpt: List[List[str]]) -> numpy.array:
        """Flattener for list of lists into a single list."""
        output = []
        for i in inpt:
            output.extend(i)
        return numpy.array(output)
    
    y_true = _list_flattener(y_true)
    y_pred = _list_flattener(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Numper of tokens don't match between y_true and y_pred.")  
    
    return numpy.divide(
      numpy.sum(numpy.char.equal(y_true, y_pred)), 
      y_true.shape[0]
    )
