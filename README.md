# The VC Method:  A Novel approach for generating realistic OHLC

In this repostiroy we are going to implement the python code for generating synthetic market data. this new approach imporve the state of the art generative techniques like GBM or TIMEGAN on several aspects like:\
-Non-false assumptions (normality assumption) \
-Clustering volatility\
-differentiated time-steps\
-No training needed\
-Faster and easier to implement\

## Funtion name:
VC_method_pandas
## Source for historical inforamtion
Yfinance library
## Required parameters:
ticker = str, Ticker of the asset that you want to simulate as expressed on yahoo finance\
number_sim = int,  Quantity of simulations\
lenght_sim = int,  Number of day for each simulation\
start_date = str,  Initial date for the historical info used for the simlations (%Y-%m-%d)\
end_date = str ,  Final date for the historical info used for the simlations  (%Y-%m-%d) \
initial_price = float  Initial price for the simulations \
## Output
-Dataframe: synthetic_data \
-Plot of synthetic data
