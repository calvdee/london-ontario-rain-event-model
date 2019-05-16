import pandas as pd
import numpy as np
from typing import Callable
from typing import Tuple

class ModelResult(object):
  __slots__ = [
    'name',
    'score',
    'stats',
    'results'
  ]

  def __init__(self, name, score, stats, results):
    self.name = name
    self.score = score
    self.stats = stats
    self.results = results  

def results_df(y: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
  """Returns a dataframe with actual, predicted, and error values.
  
  Args:
    actual: Observed values
    predicted: Predicted values

  Returns:
    A dataframe with result values
  """
  df = pd.DataFrame({'actual': y, 'predicted': y_pred}).assign(
    error=lambda x: x['predicted'] - x['actual'])
  return df

def score(errors: np.array) -> float:
  """Returns the score of the results using the `error` column.
  
  Args:
    errors: Errors computed as y - y_hat.
    
  Returns:
    The resulting score.
  """
  return np.sqrt(errors**2).mean()

def evalute_model(model_name: str, model: Callable[[pd.DataFrame], pd.Series], y: pd.Series, **args) -> Tuple:
  """Generates predictions using the model and displays the results.
  
  Args:
    name: The name of the model
    model: A model that accepts a series that it uses to make predictions.
    y: The target values.
  
  Returns: A tuple with information about the model evaluation.
  """
  y_pred = model(y, **args)
  results = results_df(y['rainEvents'], y_pred).round(0)
  
  model_score = score(results['error'])
  stats = results['error'].describe().rename(model_name)
  result = ModelResult(model_name, model_score, stats, results)

  return result