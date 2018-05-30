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
import subprocess
from scipy.optimize import curve_fit
from datetime import timedelta
import sys


# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')

from A_Data_Import import *
from B_Moment_Estimation import *
from C_Simulation import *


# -------------------- VARIABLES ----------------------------------------------------- #

input_file = 'inputs'
output_file = 'outputs'
samples = 500
branching = (2,2)
instruments = ('KO', 'MSFT', 'IBM', 'AXP', 'PG')
start_date = '2016-01-01'
end_date = '2018-01-01'
source = 'morningstar'
price_point = 'Close'
to_plot = 'yes'

# -------------------- EXECUTING C++ CODE THROUGH PYTHON ----------------------------- #

# Get the data
stock_data = import_stock_data_API(instruments=instruments, data_source= source,
                                   start_date= start_date, end_date=end_date, price_point=price_point,
                                   to_plot=to_plot)  # Takes c. 20 secs to query

# Teat exponential fit for a single stock
residuals, parameters = exponential_growth(stock_data['AXP'], to_plot=to_plot)

# Test moment-calculation
means, variances = calculate_moments(stock_data)

# Get the moments and save to format for the simulator
cpp_layout(input_file, stock_data, branching=branching)

# Run the simulation
clean_cluster()
compile_cluster() # Gives some errors, but still works
run_cluster(input_file, output_file, samples=samples)

# Use different branching
# branching = (10,10,10)
# samples = 100000
# cpp_layout(input_file, stock_data, exp_function, branching=branching)
# run_cluster(input_file, output_file, samples=samples)

# Read the output
pd.set_option('display.max_columns', 10)
scenarios = read_cluster_output(output_file, asset_names=instruments)