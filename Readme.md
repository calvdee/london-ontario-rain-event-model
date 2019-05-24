# Rain Event Analysis and Forecasting

## Motivation
The original motivation of this project was to test my hypothesis that there has been an increase in the frequency of rain events (rainy days) in London, Canada in the first five months of 2019 compared to the last 3-5 years. Of secondary interest was how the quantity of rain has changed over time, which is not necessarily correlated with how its frequency has changed. 

After understanding how the frequency of rainy days has changed over time, I wanted to put that change into the context of how we experience rainy days. My subjective view is that the frequency of rainy days is best understood and measured by the proportion of days in a week that it rains. An alternative view is that rain is best understood as the number of continuous days of rain events.

Thinking about rain event frequency in the context of weather forecasts, I realized that forecasts only ever communicate the probability of precipitation for some time period but never the number of days that it might rain. This motivated me to experiment with different predictive models to forecast the count of rain events in a seven day period starting on Mondays.