import pandas as pd
import numpy as np
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

def get_evaluation_stats(results: pd.DataFrame) -> pd.Series:
  """ Computes model result statistics.

  Args:
    results: Model evaluation results.

  Returns:
    Model statistics where each row includes a different
    statistic.
  """
  stats = results.describe()
  stats['95%'] = results.quantile(.95)
  stats['99%'] = results.quantile(.99)

  return stats