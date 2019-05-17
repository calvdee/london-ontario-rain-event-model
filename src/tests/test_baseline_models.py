import pandas as pd
from src.app.model.baseline_models import evalute_models, evalute_model, predict
from src.app.feature_data import load_feature_data
from settings import WEEKLY_DATA_PATH

identity  = lambda df: df['rainEvents'].values

def test_predict():
  y = pd.read_csv(WEEKLY_DATA_PATH)
  results = predict(identity, y)
  assert len(results) == len(y)

def test_evaluate_models():
  X = load_feature_data(WEEKLY_DATA_PATH)
  evalute_models(X)

def test_evaluate_model():
  y = pd.read_csv(WEEKLY_DATA_PATH)
  evalute_model('identity', identity, y)