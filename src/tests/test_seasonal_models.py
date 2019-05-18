import pandas as pd
from src.app.model.seasonal_models import predict
from src.app.feature_data import load_feature_data
from settings import WEEKLY_DATA_PATH

def naive(T: pd.DataFrame, seasonal_periods: int, horizon: int, **kwargs):
  last_val = T['rainEvents'].iloc[-1:]
  last_ix = last_val.index[0]

  next_val = last_val.values[0]
  next_ix = last_ix + pd.Timedelta('1W')

  return pd.Series(next_val, index=[next_ix])

def test_predict():
  X = load_feature_data(WEEKLY_DATA_PATH)
  results = predict(naive, X, seasonal_periods=2, horizon=1)
  assert len(results) == len(X)
  assert results.index[0] == X.index[0]
  assert results.index[-1] == X.index[-1]

def test_predict_hw():
  # TODO: Should sanity check the predictions made by the evalute function
  pass