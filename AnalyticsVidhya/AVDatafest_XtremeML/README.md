# Winning Solution for Analytics Vidhya Machine Learning Competition - [Xtreme ML Hack](https://datahack.analyticsvidhya.com/contest/machine-learning-hackathon/)

(c) [Sonny](https://github.com/sonnylaskar)

This model scored 60.9 on the Public Leaderboard, 61.7 on the [Private Leaderboard]("https://datahack.analyticsvidhya.com/contest/machine-learning-hackathon/lb") and ranked #2. 

## Prerequisites:
1. R version 3.3.3 
2. R Packages: readr, lubridate, dplyr, tidyr, xgboost

## Problem Statement:
The largest water supplier of Barcelona wants to leverage machine learning to effectively predict daywise-mediumwise-departmentwise breakdown of predictions of how many contacts (tickets/enquiries) would it receive and how many resolutions would it make so that they can size their team properly and improve customer satisfaction.

## Approach:
While this looked to be a time-series problem, it did not work out for me to solve it by leveraging various time series modelling techniques like ARIMA, etc. Hence I switched to solving it with regression. But the issue was that the Test dataset was in future and literally no information was known in future. We were allowed to use external data in this contest and Holiday calender seemed to be an obvious parameter that should surely affect such problems. 

### Feature Engineering:
1. Date features like weekday, quarter, etc.
2. Whether a Day was a holiday in Spain?
3. How many days were elaped since the last holiday (in rank_percent)?
4. Lagged features of # of contacts and resolutions of 75 days, 90 days and 120 days (Since the prediction to be made was upto 75 days in future, hence I decided not to include any lag value less than 75 days)

### Modeling:
Xgboost is the first model that I try everytime I have to solve any such problem. As always, it gave a significant score. For cross validation, I used the last 4 months data. 

## Steps to reproduce the submission:
1. Copy all Train files in the folder _"input/Train"_
2. Copy all Test files in the folder _"input/Test"_
3. External data: I used holiday list of Spain as an external data from [here](http://www.officeholidays.com/countries/spain/regional.php?list_year=2010&list_region=catalonia "Calender")
4. Ensure folder _"output"_ exists
5. Run the Rscript _final_model.R_ from the _code_ directory
6. The final files will be created in the _"output"_ folder

Enjoy :smile:


Regards

Sonny
