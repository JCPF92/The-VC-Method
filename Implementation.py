import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm



def VC_Method_pandas(
                    #Required parameters:
                    ticker = str, # Ticker of the asset that you want to simulate
                    number_sim = int, # Quantity of simulations
                    lenght_sim = int, # Number of day for each simulatioN
                    start_date = str, # Initial date for the historical info used for the simlations (%Y-%m-%d)
                    end_date = str , # Final date for the historical info used for the simlations  (%Y-%m-%d)
                    initial_price = float # Initial price for the simulations
                    ):
    rolling_volatility_window = 30
    
    data = yf.download(
                        ticker, 
                        start =  datetime.datetime.strptime(start_date, "%Y-%m-%d"), 
                        end = datetime.datetime.strptime(end_date, "%Y-%m-%d"), 
                        progress = False,
                        )
    
    if len(data)!=0:  
        data["Date"] = data.index
        historical_data = data[[
                                "Date",
                                 "Open",
                                 "High",
                                 "Low", 
                                 "Close",
                                 "Adj Close",]].reset_index(drop=True)

        historical_data.loc[:, "Daily fluctuation"] = ((historical_data["Close"]
                                                        -historical_data["Open"])
                                                       /historical_data["Open"]
                                                       *100)
        historical_data.loc[:, "Intra-day fluctuation"] = ((historical_data["High"]
                                                            -historical_data["Low"])
                                                           /historical_data["Open"]
                                                          *100)
        historical_data.loc[:, "Inter-day fluctuation"] = ((historical_data["Open"]
                                                            -historical_data["Close"].shift(1))
                                                           /historical_data["Close"].shift(1)
                                                          *100)

        historical_data.loc[:, "Differential of Intradf"] = (historical_data["Intra-day fluctuation"]
                                                            - abs(historical_data["Daily fluctuation"]))



        historical_data.loc[:, "Distribution of Intra-day Fluctuation"] = ((historical_data["High"]
                                                                           - historical_data["Open"])
                                                                           /(historical_data["High"]
                                                                             - historical_data["Low"]))

        historical_data['Rolling volatility'] = historical_data['Daily fluctuation'].rolling(window=rolling_volatility_window,
                                                                                            closed="left").std()

        historical_data["High volatility period"] = 0


        # Outlier Detextion IQR method for the variable:
            #  Rolling volatility
        q3, q1 = np.percentile(historical_data["Rolling volatility"].dropna(),[75, 25])
        IQR = q3-q1 
        top_limit = q3 + (1.5 * IQR)
        counter = 0
        for i in range(len(historical_data)):
            if i > rolling_volatility_window:
                if ((historical_data.loc[i, "Rolling volatility"]>= top_limit)
                    & (historical_data.loc[i-1-rolling_volatility_window:i-1, "Rolling volatility"]< top_limit).all()):

                    counter += 1
                    historical_data.loc[i-rolling_volatility_window:i, "High volatility period"] = counter

                if ((historical_data.loc[i, "Rolling volatility"]>= top_limit)
                    & (historical_data.loc[i-1, "Rolling volatility"]>= top_limit)):
                    historical_data.loc[i, "High volatility period"] = counter

                if ((historical_data.loc[i, "Rolling volatility"]< top_limit)
                    & (historical_data.loc[i-1-rolling_volatility_window:i-1, "Rolling volatility"]>= top_limit).any()):

                    historical_data.loc[i, "High volatility period"] = counter

         # Divide our sample:
            #Standard Periods
            #High Volatility periods
        standard_historical_data = historical_data[historical_data["High volatility period"]==0]
        high_volatility_historical_data = historical_data[historical_data["High volatility period"]!=0]
        if len(high_volatility_historical_data)==0:
            high_volatility_historical_data = standard_historical_data
        

        # Calculate Probability Density Functions for each variable in standar periods:
            # Daily fluctuation
            # Inter-day fluctuation
            # Differential of Intra-day fluctuation
         # Daily fluctuation (Standard)
        standard_daily_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(standard_historical_data["Daily fluctuation"].dropna())
        x = np.arange(standard_historical_data["Daily fluctuation"].min(),
                        standard_historical_data["Daily fluctuation"].max()+0.011, 0.01)
        y = (PDF_function(x))
        standard_daily_fluctuation["Values"] = x
        standard_daily_fluctuation["Proabability"] = y
        standard_daily_fluctuation["Adjusted Probability"] = (standard_daily_fluctuation["Proabability"]
                                                                / standard_daily_fluctuation["Proabability"].sum())
        # Inter-day fluctuation (Standard)
        standard_inter_day_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(standard_historical_data["Inter-day fluctuation"].dropna())
        x = np.arange(standard_historical_data["Inter-day fluctuation"].min(),
                        standard_historical_data["Inter-day fluctuation"].max()+0.011, 0.01)
        y = (PDF_function(x))
        standard_inter_day_fluctuation["Values"] = x
        standard_inter_day_fluctuation["Proabability"] = y
        standard_inter_day_fluctuation["Adjusted Probability"] = (standard_inter_day_fluctuation["Proabability"]
                                                                / standard_inter_day_fluctuation["Proabability"].sum())
        # Differential of Intra-day fluctuation (Standard)
        standard_diff_intradf_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(standard_historical_data["Differential of Intradf"].dropna())
        x = np.arange(standard_historical_data["Differential of Intradf"].min(),
                        standard_historical_data["Differential of Intradf"].max()+0.011, 0.01)
        y = (PDF_function(x))
        standard_diff_intradf_fluctuation["Values"] = x
        standard_diff_intradf_fluctuation["Proabability"] = y
        standard_diff_intradf_fluctuation["Adjusted Probability"] = (standard_diff_intradf_fluctuation["Proabability"]
                                                                    / standard_diff_intradf_fluctuation["Proabability"].sum())
    
        # Calculate Probability Density Functions for each variable during High Volatility periods:
            # Daily fluctuation
            # Inter-day fluctuation
            # Differential of Intra-day fluctuation

        # Daily fluctuation (High volatility)
        HV_daily_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(high_volatility_historical_data["Daily fluctuation"].dropna())
        x = np.arange(high_volatility_historical_data["Daily fluctuation"].min(),
                        high_volatility_historical_data["Daily fluctuation"].max()+0.011,
                      0.01)
        y = (PDF_function(x))
        HV_daily_fluctuation["Values"] = x
        HV_daily_fluctuation["Proabability"] = y
        HV_daily_fluctuation["Adjusted Probability"] = (HV_daily_fluctuation["Proabability"]
                                                        / HV_daily_fluctuation["Proabability"].sum())
        # Inter-day fluctuation (High volatility)
        HV_inter_day_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(high_volatility_historical_data["Inter-day fluctuation"].dropna())
        x = np.arange(high_volatility_historical_data["Inter-day fluctuation"].min(),
                        high_volatility_historical_data["Inter-day fluctuation"].max()+0.011,
                      0.01)
        y = (PDF_function(x))
        HV_inter_day_fluctuation["Values"] = x
        HV_inter_day_fluctuation["Proabability"] = y
        HV_inter_day_fluctuation["Adjusted Probability"] = (HV_inter_day_fluctuation["Proabability"]
                                                            / HV_inter_day_fluctuation["Proabability"].sum())
        # Differential of Intra-day fluctuation (High volatility)
        HV_diff_intradf_fluctuation = pd.DataFrame()
        PDF_function = stats.kde.gaussian_kde(high_volatility_historical_data["Differential of Intradf"].dropna())
        x = np.arange(high_volatility_historical_data["Differential of Intradf"].min(),
                      high_volatility_historical_data["Differential of Intradf"].max()+0.011,
                      0.01)
        y = (PDF_function(x))
        HV_diff_intradf_fluctuation["Values"] = x
        HV_diff_intradf_fluctuation["Proabability"] = y
        HV_diff_intradf_fluctuation["Adjusted Probability"] = (HV_diff_intradf_fluctuation["Proabability"]
                                                                / HV_diff_intradf_fluctuation["Proabability"].sum())

        # Calculate the Probability Density Function for the length of High Volatility periods
        length_HV_periods = pd.DataFrame()
        if len(high_volatility_historical_data["High volatility period"].value_counts())>1:

            distribucion_duracion = stats.kde.gaussian_kde(high_volatility_historical_data["High volatility period"].value_counts())
            x = np.arange(high_volatility_historical_data["High volatility period"].value_counts().min(),
                          high_volatility_historical_data["High volatility period"].value_counts().max())
            y = (distribucion_duracion(x))
            length_HV_periods["Values"] = x
            length_HV_periods["Proabability"] = y
            length_HV_periods["Adjusted Probability"] =  (length_HV_periods["Proabability"]
                                                          / length_HV_periods["Proabability"].sum())

        else:

            x = range(rolling_volatility_window,
                      high_volatility_historical_data["High volatility period"].value_counts().max())
            y = 100/len(x)
            length_HV_periods["Values"] = x
            length_HV_periods["Proabability"] = y
            length_HV_periods["Adjusted Probability"] = (length_HV_periods["Proabability"]
                                                         / length_HV_periods["Proabability"].sum())


        #Generate the Linear Regression Model:
            #Dependent variable = Daily fluctuation
            # Independent variable = Distribution of Intra-day Fluctuation
        x = historical_data["Daily fluctuation"]
        x = sm.add_constant(x)
        y = historical_data["Distribution of Intra-day Fluctuation"]
        white_noise = np.arange(-0.05 , 0.05, 0.01)
        linear_regression_model = sm.OLS(endog=y, exog=x)
        linear_regression_model = linear_regression_model.fit()

        #Create the chart for plotting synthetic data
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.grid(True, axis='y')
        ax.grid(True, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.suptitle(f"Prices evolution")
        ax.set_xlabel('days')
        ax.set_ylabel('$')

        days = range(lenght_sim)
        simulation_counter = 1
        synthetic_data = pd.DataFrame()
        for sim in range(number_sim):

            simulation = pd.DataFrame(columns = [
                                                 f"Open {sim}",
                                                 f"High {sim}",
                                                 f"Low {sim}", 
                                                 f"Close {sim}",
                                                 f"Daily fluctuation {sim}",
                                                 f"Differential of Intradf {sim}",
                                                 f"Intra-day fluctuation {sim}",
                                                 f"Distribution of Intra-day Fluctuation {sim}",
                                                 f"Inter-day fluctuation {sim}",
                                                 f"High volatility period {sim}",
                                                 f"Max. Drawdown {sim}"])




            HV_counter = 0
            HV_days = ()
            for day in days :
                # Replication of stochastic process for starting a High volatility period and its length
                HV_starting_probability = (historical_data["High volatility period"].max()
                                           /len(historical_data))
                starting_high_volatility = np.random.choice([0,1], p = [1-HV_starting_probability,
                                                                        HV_starting_probability])
                if starting_high_volatility == 1:

                    HV_length = np.random.choice(length_HV_periods["Values"],
                                                p= length_HV_periods["Adjusted Probability"])

                    HV_days = range(day,
                                    day+HV_length)
                    HV_counter += 1
                # Replication of Stochastic process using High Volatility periods PDF´s
                if day in HV_days:

                    simulation.loc[day,f"Daily fluctuation {sim}"] = np.random.choice(HV_daily_fluctuation["Values"],
                                                                              p = HV_daily_fluctuation["Adjusted Probability"])
                    simulation.loc[day,f"Differential of Intradf {sim}"] = np.random.choice(HV_diff_intradf_fluctuation["Values"],
                                                                                     p = HV_diff_intradf_fluctuation["Adjusted Probability"])
                    simulation.loc[day,f"Intra-day fluctuation {sim}"] = abs(simulation.loc[day,f"Daily fluctuation {sim}"]
                                                                      + (simulation.loc[day,f"Differential of Intradf {sim}"]
                                                                         * np.sign(simulation.loc[day,f"Daily fluctuation {sim}"])))
                    simulation.loc[day,f"Inter-day fluctuation {sim}"] = np.random.choice(HV_inter_day_fluctuation["Values"],
                                                                                   p = HV_inter_day_fluctuation["Adjusted Probability"])
                    predict = pd.DataFrame()
                    predict.loc[0,"Daily fluctuation"] = simulation.loc[day,f"Daily fluctuation {sim}"]
                    predict = sm.add_constant(predict, has_constant = "add")
                    simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"]= np.clip(linear_regression_model.predict(predict)[0]
                                                                                        + np.random.choice(white_noise),
                                                                                        0 , 1)
                    simulation.loc[day,f"High volatility period {sim}"] = contador
                # Replication of Stochastic process using Standard periods PDF´s
                else:

                    simulation.loc[day,f"Daily fluctuation {sim}"] = np.random.choice(standard_daily_fluctuation["Values"], 
                                                                               p = standard_daily_fluctuation["Adjusted Probability"])
                    simulation.loc[day,f"Differential of Intradf {sim}"] = np.random.choice(standard_diff_intradf_fluctuation["Values"],
                                                                                    p = standard_diff_intradf_fluctuation["Adjusted Probability"])
                    simulation.loc[day,f"Intra-day fluctuation {sim}"] = abs(simulation.loc[day,f"Daily fluctuation {sim}"]
                                                                      + (simulation.loc[day,f"Differential of Intradf {sim}"]
                                                                         * np.sign(simulation.loc[day,f"Daily fluctuation {sim}"])))
                    simulation.loc[day,f"Inter-day fluctuation {sim}"] = np.random.choice(standard_inter_day_fluctuation["Values"],
                                                                                   p = standard_inter_day_fluctuation["Adjusted Probability"])
                    predict = pd.DataFrame()
                    predict.loc[0,"Daily fluctuation"] = simulation.loc[day,f"Daily fluctuation {sim}"]
                    predict = sm.add_constant(predict, has_constant = "add")
                    simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"]= np.clip(linear_regression_model.predict(predict)[0]
                                                                                        + np.random.choice(white_noise),
                                                                                        0 , 1)
                    simulation.loc[day,f"High volatility period {sim}"] = 0

                # Controls for assuring Internal Consistency:   
                if (simulation.loc[day,f"Intra-day fluctuation {sim}"]
                    * simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"]
                    < simulation.loc[day,f"Daily fluctuation {sim}"]):

                    simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"] = 1

                if (simulation.loc[day,f"Intra-day fluctuation {sim}"]
                    * -(1 - simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"])
                    > simulation.loc[day,f"Daily fluctuation {sim}"]):

                    simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"] = 0

                # Construction of prices and other variables:   
                if day == 0 : 
                    simulation.loc[0,f"Open {sim}"] = initial_price 
                else:
                    simulation.loc[day,f"Open {sim}"] = (simulation.loc[day-1,f"Close {sim}"]
                                                  *(1+simulation.loc[day,f"Inter-day fluctuation {sim}"]
                                                    /100))

                simulation.loc[day,f"Close {sim}"] = (simulation.loc[day,f"Open {sim}"]
                                               *(1+simulation.loc[day,f"Daily fluctuation {sim}"]
                                                 /100))

                simulation.loc[day,f"High {sim}"] = (simulation.loc[day,f"Open {sim}"] + 
                                              (simulation.loc[day,f"Open {sim}"] 
                                               * simulation.loc[day,f"Intra-day fluctuation {sim}"]/100
                                               *simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"]))

                simulation.loc[day,f"Low {sim}"] = (simulation.loc[day,f"Open {sim}"] - 
                                              (simulation.loc[day,f"Open {sim}"] 
                                               * simulation.loc[day,f"Intra-day fluctuation {sim}"]/100
                                               *(1-simulation.loc[day,f"Distribution of Intra-day Fluctuation {sim}"])))

                simulation.loc[day,f"Max. Drawdown {sim}"] =  round(min(((simulation.loc[day, f"Close {sim}"] )
                                                             / (simulation.loc[:day, f"Close {sim}"].max())
                                                             -1 ),0),4)

            simulation = simulation.infer_objects()

            ax.plot(simulation[f"Close {sim}"])
            synthetic_data = pd.concat([synthetic_data, 
                                        simulation],
                                       axis=1)
    
        return synthetic_data
    
