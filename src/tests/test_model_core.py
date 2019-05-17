import pandas as pd
from src.app.model.core import evalute_model, predict, get_evaluation_stats
from settings import WEEKLY_DATA_PATH

identity  = lambda df: df['rainEvents'].values

def test_predict():
  y = pd.read_csv(WEEKLY_DATA_PATH)
  results = predict(identity, y)
  assert len(results) == len(y)

def test_get_stats():
  y = pd.read_csv(WEEKLY_DATA_PATH)
  results = predict(identity, y)
  stats = get_evaluation_stats(results)

def test_evaluate_model():
  y = pd.read_csv(WEEKLY_DATA_PATH)
  evalute_model('identity', identity, y)