"""
Project:
    Imperial College London MRes Dissertation Project (2017-18)

Description:
    We consider a robust multi-stage portfolio framework for optimising conditional value at risk (CVaR). During the
    first-stage, robustness is ensured by considering an ellipsoid uncertainty set for the corresponding expected
    returns. Subsequent stages are modelled using a robust framework with a (discrete) set of rival scenario trees.
    The first-stage essentially leads to a semi-infinite optimisation problem that can be simplified using duality
    to a second-order cone program. The discrete set of rival scenario trees in subsequent stages lead to discrete
    minimax. For large-scale scenario trees, Benders decomposition is needed. Past backtests with rival scenario
    trees have consistently yielded favourable results. The project will implement and test the basic model and
    subsequently consider its variants.

Authors:
  Sven Sabas

Date:
  23/05/2018
"""

# ------------------------------ IMPORT LIBRARIES --------------------------------- #

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas_datareader import data as datareader
import pickle
import requests
import bs4 as bs


# -------------------- IMPORT STOCK-PRICE TIME SERIES  ----------------------------- #

# From Morningstar through Python API
def import_stock_data_api(instruments=('KO', 'F', 'IBM', 'AXP', 'PG'), data_source='morningstar',
                          start_date= '1980-01-01', end_date='2018-01-01', price_point='Close',
                          random='no', number=10, random_seed=500, to_plot='yes', to_save='no',
                          from_file='yes', frequency='daily', folder=''):

    # Choose random stocks and pull from database
    if from_file == 'yes':

        price_series = pd.read_csv(os.getcwd() + '/data/morningstar/data_for_%s_stocks.csv' %(str(len(instruments))),
                                   header=0, index_col=0, parse_dates=True)

    else:

        if random == 'yes':

            # Get tickers
            SP500_tickers = get_SP500_tickers()

            # Choose random tickers
            np.random.seed(random_seed)
            instruments = np.random.choice(SP500_tickers, number, replace=False)

            # Make sure instruments are in capital letters
            instruments = [instrument.upper() for instrument in instruments]

            # Retrieve the stock prices
            panel_data = datareader.DataReader(instruments, data_source, start_date, end_date)

        # Pull historical data of predefined stocks
        else:


            # Make sure instruments are in capital letters
            instruments = [instrument.upper() for instrument in instruments]

            panel_data = datareader.DataReader(instruments, data_source, start_date, end_date, retry_count=0)

        # Format output
        price_series = panel_data[price_point]
        price_series = price_series.unstack(level=-2)

        # Getting all weekdays between start date and end date
        all_weekdays = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex using all_weekdays as the new index
        price_series = price_series.reindex(all_weekdays)

        # Reindexing will insert missing values (NaN) for the dates that were not present
        # in the original set. To cope with this, we can fill the missing by replacing them
        # with the latest available price for each instrument.
        price_series = price_series.fillna(method='ffill')

    if frequency == 'weekly':

        price_series = price_series.resample('W-MON').first()#, convention='start').first()

    if to_plot == 'yes':

        plt.style.use("seaborn-darkgrid")
        ax = price_series.plot(legend=True, title='Time-series of Stock Prices', colormap='ocean', figsize=(9,6))
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Closing Price ($)")
        plt.tight_layout()

    if to_save == 'yes':

        if not os.path.exists(os.getcwd() + '/results/' + folder):
                os.makedirs(os.getcwd() + '/results/' + folder)

        price_series.to_csv(os.getcwd() + '/results/' + folder + '/stock_data.csv')
        plt.savefig(os.getcwd() + '/results/' + folder + '/stock_data.pdf')

    return price_series


# From saved files
def import_data_saved(instrument_type='stocks', instruments=('ko', 'f', 'ibm', 'axp', 'pg'),
                      random='yes', price='Close', number=10, random_seed=500, remove_NA='yes',
                      to_plot='yes'):

    # Set random seed
    np.random.seed(random_seed)

    # Specify the directory path to the right instruments
    if instrument_type == 'stocks':
        path = os.getcwd() + '/data/Stocks'

    else:
        path = os.getcwd() + '/data/ETFs'

    # Extract the right instrument(s)
    if random == 'yes':

        # Number of assets
        assets = len(os.listdir(path))
        random_assets = np.random.choice(assets, number, replace=False)

        # Extract assets
        k = 0
        for i in random_assets:

            if k == 0:
                price_series = pd.read_csv(path + "/" + os.listdir(path)[i])
                price_series = price_series.set_index('Date')
                price_series.index = pd.to_datetime(price_series.index)
                price_series = price_series[price]
                price_series.name = os.listdir(path)[i].split('.')[0]
                merged_price_series = price_series

                k += 1

            else:

                price_series = pd.read_csv(path + "/" + os.listdir(path)[i])
                price_series = price_series.set_index('Date')
                price_series.index = pd.to_datetime(price_series.index)
                price_series = price_series[price]
                price_series.name = os.listdir(path)[i].split('.')[0]
                merged_price_series = pd.concat((merged_price_series, price_series), axis=1)

    else:

        k = 0
        for i in instruments:

            if k == 0:
                price_series = pd.read_csv(path + "/" + i + ".us.txt")
                price_series = price_series.set_index('Date')
                price_series.index = pd.to_datetime(price_series.index)
                price_series = price_series[price]
                price_series.name = i
                merged_price_series = price_series

                k += 1

            else:

                price_series = pd.read_csv(path + "/" + i + ".us.txt")
                price_series = price_series.set_index('Date')
                price_series.index = pd.to_datetime(price_series.index)
                price_series = price_series[price]
                price_series.name = i
                merged_price_series = pd.concat((merged_price_series, price_series), axis=1)


    # Remove NAs
    if remove_NA == 'yes':
        index = merged_price_series[merged_price_series.isnull().any(axis=1)].tail(1).index + timedelta(days=1)
        merged_price_series = merged_price_series.loc[index.strftime("%Y-%m-%d")[0]:]

    # Sense-check with plotting
    if to_plot == 'yes':
        merged_price_series.plot(legend=True)

    return merged_price_series


# Get SP500 tickers
def get_SP500_tickers():

    # Get ticker names
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

####################### END ########################