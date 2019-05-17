import pandas as pd
from src.app.model.core import get_evaluation_stats
from settings import WEEKLY_DATA_PATH

def test_get_stats():
  results = pd.Series([1, 0, 1])
  stats = get_evaluation_stats(results)