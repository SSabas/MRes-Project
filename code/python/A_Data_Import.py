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
from datetime import timedelta

# -------------------- IMPORT STOCK-PRICE TIME SERIES ----------------------------- #


def import_data(instrument_type='stocks', instruments=['ko', 'f', 'ibm', 'axp', 'pg'],
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

####################### END ########################