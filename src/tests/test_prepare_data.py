from src.app.data import get_source_data, compute_daily_averages, add_columns

RAW_DATA_PATH = 'data/raw/scrapy-weather-data.csv'

def test_get_data():
  df = get_source_data(RAW_DATA_PATH)
  assert len(df) > 0

def test_compute_daily_averages():
  source_df = get_source_data(RAW_DATA_PATH)
  avg_df = compute_daily_averages(source_df)
  assert len(avg_df) < len(source_df)
  assert len(avg_df.columns) < len(source_df.columns)

def test_add_columns():
  source_df = get_source_data(RAW_DATA_PATH)
  avg_df = compute_daily_averages(source_df)
  final_df = add_columns(avg_df)
  assert len(final_df.columns) > len(avg_df.columns)