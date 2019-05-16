from .base import ModelResult
import matplotlib.pyplot as plt

def display_results(result: ModelResult) -> None:
  """Displays the model's score, results over time, and error distribution.
  
  Args:
    results: A model's results.
  """
  results = result.results

  print(f'{result.name} Score:', result.score)

  plt.figure(figsize=(14,5))
  results['actual'].plot(label='actual');
  fig = results['predicted'].plot(label='predicted', style='-');
  fig.set_title('Actual vs. Forecast')
  plt.legend();

  plt.figure(figsize=(14,5))
  fig = results['error'].plot();
  fig.set_title('Error')
  
  plt.figure()
  fig = results['error'].hist();
  fig.set_title('Error Distribution')
    
  plt.show()

  stats = results['error'].describe().rename(result.name)
  print('Error stats:')
  print(stats)