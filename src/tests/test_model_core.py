import pandas as pd
from src.app.model.core import evalute_model
from settings import WEEKLY_DATA_PATH

def test_evaluate_model():
  identity  = lambda df: df['rainEvents'].values
  y = pd.read_csv(WEEKLY_DATA_PATH)
  evalute_model('identity', identity, y)

