from src.app.feature_data import prepare_weekly_rain_day_counts
from src.app.source_data import prepare_daily_data
from settings import RAW_DATA_PATH

def test_prepare_weekly_rain_day_counts(tmp_path):
  output_path = tmp_path / 'daily_data.csv'
  daily_data = prepare_daily_data(RAW_DATA_PATH)
  daily_data.to_csv(output_path)
  counts = prepare_weekly_rain_day_counts(output_path)

  print(counts.head(1))

  assert len(counts) > 0