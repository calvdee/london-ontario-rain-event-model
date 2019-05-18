import pandas as pd

from typing import Callable
from .core import results_df, get_evaluation_stats, score, ModelResult

def predict(
  model: Callable[[pd.DataFrame, int, int], pd.Series], 
  X: pd.DataFrame, 
  seasonal_periods: int,
  horizon: int, 
  **kwargs) -> pd.DataFrame:
  """ Generates model walk-forward model predictions over by executing 
  the Callable, passing `X` and `kwargs`. Assumes that the model function
  returns a pandas.Series the with length `horizon`.

  Args:
    model: A function that accepts a dataframe, `seasonal_periods` and `horizon`
           and returns a sequence of predictions.
    X: Observed values.
    seasonal_periods: The number of periods in a season.
    horizon: The forecast horizon (number of forecasts to generate)

  Returns:
    Model prediction results.
  """

  y_hats = []

  for ix in range(seasonal_periods, len(X)):
    y = X['rainEvents']

    # We need at least `period` observations to estimate the seasonal component
    # plus `ix` observations to estimate level and trend
    T = X.iloc[0:ix, :]
    fcst = model(T, seasonal_periods, horizon, **kwargs)
    y_hats.append(fcst)

  # We will be missing `seasonal_periods` - add the missing values to
  # align the series
  fst_ix = X.sort_index().index[0]
  missing = pd.Series(index=pd.date_range(
    start=fst_ix, freq='W-MON', closed='left', periods=seasonal_periods))
  
  # Concatenate the individual series
  y_hats = pd.concat(y_hats)

  # Add the missing data
  y_hats_all = pd.concat([missing, y_hats])

  results = results_df(X['rainEvents'], y_hats_all)

  return results

def evalute_model(
  model_name: str,   
  model: Callable[[pd.DataFrame, int, int], pd.Series], 
  X: pd.DataFrame, 
  seasonal_periods: int,
  horizon: int, 
  **kwargs) -> ModelResult:
  """Generates predictions using the model and displays the results.
  
  Args:
    name: The name of the model.
    model: A function that accepts a dataframe, `seasonal_periods` and `horizon`
           and returns a sequence of predictions.
    X: Observed values.
    seasonal_periods: The number of periods in a season.
    horizon: The forecast horizon (number of forecasts to generate)
  
  Returns: A ModelResult object.
  """
  results = predict(model, X, seasonal_periods, horizon, **kwargs)
  
  model_score = score(results['error'])
  stats = get_evaluation_stats(results['error']).rename(model_name)
  result = ModelResult(model_name, model_score, stats, results)

  return result