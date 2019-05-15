from src.app.source_data import _get_source_data, _compute_daily_averages, _add_columns
from settings import RAW_DATA_PATH

def test_get_data():
  df = _get_source_data(RAW_DATA_PATH)
  assert len(df) > 0

def test_compute_daily_averages():
  source_df = _get_source_data(RAW_DATA_PATH)
  avg_df = _compute_daily_averages(source_df)
  assert len(avg_df) < len(source_df)
  assert len(avg_df.columns) < len(source_df.columns)

def test_add_columns():
  source_df = _get_source_data(RAW_DATA_PATH)
  avg_df = _compute_daily_averages(source_df)
  new_df = _add_columns(avg_df)
  assert len(new_df.columns) > len(avg_df.columns)