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

  with mlflow.start_run(run_name='naive'):
    naive_results = evalute_model('Naive', predict_naive, X)
    print('Naive score:', naive_results.score)
    mlflow.log_metric('rmse', naive_results.score)

  with mlflow.start_run(run_name='seasonal_naive'):
    snaive_results = evalute_model('Seasonal Naive', predict_snaive, X)
    print('Seasonal naive score:', snaive_results.score)
    mlflow.log_metric('rmse', snaive_results.score)

  with mlflow.start_run(run_name='moving_average'):
    best_k = optimize_ma_k(X)
    ma_results = evalute_model('Moving Average', predict_ma, X)
    print(f'MA score (k={best_k}):', ma_results.score)
    mlflow.log_param('k', best_k)
    mlflow.log_metric('rmse', ma_results.score)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  # parser.add_argument('output')
  args = parser.parse_args()
  

  print(f'Evaluating models with data from file {args.input}')

  X = load_feature_data(args.input)
  evalute_models(X)

  # output = prepare_weekly_rain_day_counts(args.input)

  # output.to_csv(args.output, index=False, sep=',')

  # print(f'Saved file to {args.output}')

