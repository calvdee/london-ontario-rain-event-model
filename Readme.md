# Rain Event Analysis and Forecasting

## Motivation
The original motivation of this project was to explore how the frequency of rain events (rainy days) in my hometown and current city, London, Canada, has changed over the last ten years. I was interested in framing rainy day frequency in the context of weeks rather than looking at single number like the raw frequency or proportion  of rainy days over the period because my (subjective) view is that the frequency of rainy days is best conceptualized and communicated by the number of days in a week that it rains. An alternative view is that rain is best understood as the number of continuous days of rain events which presents an opportunity for a different analysis.

Thinking about rain event frequency in the context of weather forecasts, I realized that forecasts only ever communicate the probability of precipitation for a given time period but never the number of days that it might rain. This motivated me to experiment with different predictive models to directly forecast the count of rain events in a seven day period starting on Mondays, comparing the performance to at least one weather forecasting service.