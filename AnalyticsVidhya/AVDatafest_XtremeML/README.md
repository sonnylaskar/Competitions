# Winning Solution for Analytics Vidhya Machine Learning Competition - [Xtreme ML Hack](https://datahack.analyticsvidhya.com/contest/machine-learning-hackathon/)

(c) [Sonny](https://github.com/sonnylaskar)

This model scored 60.9 on the Public Leaderboard, 61.7 on the Private Leaderboard and ranked #2

## Prerequisites:
1. R version 3.3.3 
2. R Packages: readr, lubridate, dplyr, tidyr, xgboost
3. Any Linux Distribution

## Steps:
1. Copy all Train files in the folder "input/Train"
2. Copy all Test files in the folder "input/Test"
3. External data:
	I used holiday list of Spain as an external data since it is readily available anytime
	I manually created the list from the this [url]("http://www.officeholidays.com/countries/spain/regional.php?list_year=2010&list_region=catalonia" "Calender")
	Note: By changing the year, we can get for each year from 2010 to 2017
	The holiday list is saved in "input/holiday.csv"
4. Ensure folder "output" exists
5. Execute the below command:
	$ cd code
	$ Rscript final_model.R
6. The final files will be created in the "output" folder
Enjoy :-)


Regards

Sonny
