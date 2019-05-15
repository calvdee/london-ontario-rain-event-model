import pandas as pd
import argparse

def _season(date):
  if date.month == 12 or date.month < 3:
    # December - Feb
    return 'winter'
  elif date.month >= 3 and date.month < 6:
    # March - June
    return 'spring'
  elif date.month >= 6 and date.month < 9:
    # June - Sept
    return 'summer'
  else:
    # Sept - Nov
    return 'fall'

def _get_source_data(data_path: str) -> pd.DataFrame():
  """ Retrieves source data from the `data_path`. The source data
  is the data retrieved directly from the historical weather data
  provider.
  """
  df = pd.read_csv(data_path, parse_dates=['date'])

  # Discrete columns
  disc = ['station', 'date']

  # Continuous data with empty values replaced with `NaN` and '<31' with 31
  df_cont = pd.DataFrame.replace(
    pd.DataFrame(df[df.columns.drop(disc)]),
    ["\xa0", "<31"],
    ["NaN", 31]).astype(float)
  df_disc = df[disc]
  df = pd.concat([df_cont, df_disc], axis=1)

  return df

def _compute_daily_averages(source_data: pd.DataFrame) -> pd.DataFrame():
  """ Aggregates relevent weather data from multiple weather stations."""
  cols = [
    'date',
    'totalPrecipMM', 
    'minTemp', 
    'maxTemp', 
    'meanTemp', 
    'totalSnowCM'
  ]
  df = source_data[cols].set_index('date').astype(float).resample('1D').mean()
  return df

def _add_columns(data: pd.DataFrame) -> pd.DataFrame:
  """ Adds additional columns to the `data`.
  """
  new_data = data.copy()
  # Season
  new_data['season'] = pd.Series(data.index).apply(_season).values

  # Dates
  new_data['year'] = pd.Series(data.index).apply(lambda x: x.year).values
  new_data['month'] = pd.Series(data.index).apply(lambda x: x.month).values

  return new_data

def prepare_daily_data(data_path: str) -> pd.DataFrame():
  """ Runs the required steps to transform raw weather observations
  from multiple stations into daily weather observations - one record
  per day.
  """
  source = _get_source_data(data_path)
  avg = _compute_daily_averages(source)
  daily = _add_columns(avg)
  return daily


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('output')
  args = parser.parse_args()

  print(f'Processing input file {args.input}')

  output = prepare_daily_data(args.input)

  output.to_csv(args.output, index=True, sep=',')

  print(f'Saved file to {args.output}')

  