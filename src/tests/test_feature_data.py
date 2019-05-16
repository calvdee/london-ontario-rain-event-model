import pandas as pd
from src.app.feature_data import prepare_weekly_rain_day_counts, save_feature_data, load_feature_data
from src.app.source_data import prepare_daily_data
from settings import DAILY_DATA_PATH, WEEKLY_DATA_PATH

def test_prepare_weekly_rain_day_counts(tmp_path):
  counts = prepare_weekly_rain_day_counts(DAILY_DATA_PATH)

  print(counts.head(1))

  assert len(counts) > 0

def test_save_weekly_rain_day_counts(tmp_path):
  output_path = tmp_path / 'weekly_data.csv'
  counts = prepare_weekly_rain_day_counts(DAILY_DATA_PATH)
  
  path = save_feature_data(counts, output_path)
  path_counts = pd.read_csv(path)
  assert len(path_counts) > 0

def test_load_weekly_rain_day_counts():
  counts = prepare_weekly_rain_day_counts(DAILY_DATA_PATH).set_index('date')
  file_counts = load_feature_data(WEEKLY_DATA_PATH)

  assert len(file_counts) > 0
  assert all(x == y for x, y in zip(counts.columns, file_counts.columns))