import pandas as pd
import argparse

def prepare_weekly_rain_day_counts(source_path: str) -> pd.DataFrame:
  """ Computes the number of rainy days per week. Expects the data at
  `source_path` to include daily weather observations.
  """
  source = pd.read_csv(source_path, parse_dates=['date']).set_index('date')
  rain_days = source.query('totalPrecipMM >= 0.2')
  rain_counts = (
    rain_days
      .resample('W-MON', label='left', closed='left')
      .size()
      .rename('rainEvents')
      .reset_index()
  )

  return rain_counts

def save_feature_data(data: pd.DataFrame, feature_path: str) -> str:
  """ Saves the data to the `feature_path`.
  """
  data.to_csv(feature_path, index=False, sep=',')
  return feature_path

def load_feature_data(feature_path: str) -> pd.DataFrame:
  """ Loads the data from `feature_path`.
  """
  # Putting the `get` function inside the same module as the `prepare` function
  # helps align reading and writing the data (using e.g. the same arguments to 
  # the CSV method)
  data = pd.read_csv(feature_path, parse_dates=['date']).set_index('date')
  return data

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  args = parser.parse_args()

  print(f'Processing input file {args.input}')

  output = prepare_weekly_rain_day_counts(args.input)

  save_feature_data(output, args.output)

  print(f'Saved file to {args.output}')