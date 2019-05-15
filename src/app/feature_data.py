import pandas as pd

def prepare_weekly_rain_day_counts(source_path: str) -> pd.DataFrame:
  """ Computes the number of rainy days per week. Expects the data at
  `source_path` to include daily weather observations.
  """
  source = pd.read_csv(source_path, parse_dates=['date']).set_index('date')
  rain_days = source.query('totalPrecipMM >= 0.2')
  rain_counts = rain_days.resample('1W', label='left').size()

  return rain_counts