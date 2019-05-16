import argparse
import pandas as pd
import numpy as np
import mlflow

from .core import evalute_model, score
from ..feature_data import load_feature_data

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

def evalute_models(X: pd.DataFrame):
  experiment_id = mlflow.set_experiment('baseline_models')

  models = zip(
    ['naive', 'seasonal_naive', 'moving_average'],
    [predict_naive, predict_snaive, predict_ma])

  best_model = ('',1e10)

  for name, model in models:
    with mlflow.start_run(run_name=name):
      result = evalute_model(name, model, X)
      # print(f'{name} score:', result.score)
      mlflow.log_metric('rmse', result.score)

      if result.score < best_model[1]:
        best_model = (name, result.score)

  return best_model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  # parser.add_argument('output')
  args = parser.parse_args()
  

  # print(f'Evaluating models with data from file {args.input}')

  X = load_feature_data(args.input)
  best_model = evalute_models(X)

  print(f'Best model: {best_model[0]}, RMSE: {best_model[1].round(3)}')
  # output = prepare_weekly_rain_day_counts(args.input)

  # output.to_csv(args.output, index=False, sep=',')

  # print(f'Saved file to {args.output}')

