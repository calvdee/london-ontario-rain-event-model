import argparse
import pandas as pd
import numpy as np
import mlflow

from .core import score, results_df, get_evaluation_stats, ModelResult
from ..feature_data import load_feature_data
from typing import Callable

def predict(model: Callable[[pd.DataFrame], pd.Series], x: pd.DataFrame, **args) -> pd.DataFrame:
  """ Generates model predict by executing the Callable with `x` and args.

  Args:
    model: A function that accepts a dataframe and returns a
           sequence of predictions.
    x: Observed values.

  Returns:
    Model prediction results.
  """
  y_pred = model(x, **args)
  results = results_df(x['rainEvents'], y_pred).round(0)

  return results

def predict_naive(X):
  return X.iloc[:,0].shift(1)

def predict_snaive(X):
  return X.iloc[:,0].shift(52)

def predict_ma(X: pd.DataFrame, k=2):
  # shift(1) excludes the current instance from being included
  # in the MA
  return X.iloc[:,0].shift(1).rolling(k).mean().fillna(0).astype(int)

def optimize_ma_k(weekly_counts):
  ma_results = []
  ks = np.arange(2, 31)

  for k in ks:
    # print(f'Building MA model, k={k}', end=' ')
    preds = predict_ma(weekly_counts, k)
    error = weekly_counts.iloc[:,0] - preds
    model_score = score(error)
    ma_results.append(model_score)
    # print(f'Score={model_score}')

  best_k = ks[np.argmin(ma_results)]

  return best_k

def evalute_model(model_name: str, model: Callable[[pd.DataFrame], pd.Series], y: pd.Series, **args) -> ModelResult:
  """Generates predictions using the model and displays the results.
  
  Args:
    name: The name of the model.
    model: A model that accepts a series that it uses to make predictions.
    y: The target values.
  
  Returns: A tuple with information about the model evaluation.
  """
  results = predict(model, y)
  
  model_score = score(results['error'])
  stats = get_evaluation_stats(results['error']).rename(model_name)
  result = ModelResult(model_name, model_score, stats, results)

  return result

def evalute_models(X: pd.DataFrame) -> ModelResult:
  experiment_id = mlflow.set_experiment('baseline_models')

  models = zip(
    ['naive', 'seasonal_naive', 'moving_average'],
    [predict_naive, predict_snaive, predict_ma])

  best_result = ModelResult('', 1e10, None, None)

  for name, model in models:
    with mlflow.start_run(run_name=name):
      result = evalute_model(name, model, X)
      # print(f'{name} score:', result.score)
      mlflow.log_metric('rmse', result.score)

      if result.score < best_result.score:
        best_result = result

  return best_result

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  # parser.add_argument('output')
  args = parser.parse_args()
  

  # print(f'Evaluating models with data from file {args.input}')

  X = load_feature_data(args.input)
  best_model = evalute_models(X)

  print(best_model.name)
  print('RMSE:', best_model.score.round(3))
  print(best_model.stats[['mean', '50%', '95%', '99%', 'max']])
  # output = prepare_weekly_rain_day_counts(args.input)

  # output.to_csv(args.output, index=False, sep=',')

  # print(f'Saved file to {args.output}')

