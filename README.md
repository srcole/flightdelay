# flightdelay
Analysis of delays of US flights in 2015.

# Summary
For a class project ([CSE 258](http://cseweb.ucsd.edu/classes/wi17/cse258-a/) at UCSD), we explored statistics on US domestic flights and created a model to predict if a flight would be delayed by 15 or more minutes.

# Analysis
* LIST NOTEBOOKS

# Data acquisition and cleaning
Download data that includes information on all domestic flights in 2015
from [Kaggle](https://www.kaggle.com/usdot/flight-delays).

After acquired, the data can be cleaned as in [this notebook](https://github.com/srcole/flightdelay/blob/master/nbsc/1b_Resave%20flights%20data%20for%20all%20flights%20to%20use%203digit%20airport%20codes.ipynb) to make the airport codes consistent, and [this notebook](https://github.com/srcole/flightdelay/blob/master/nbsc/1d_Replace%20BSM%20with%20AUS.ipynb) to fix the inconsistency in the Austin airport encoding.

# fld
Some tools for data loading, exploration, etc.

# nbs_development
Notebooks written during the project