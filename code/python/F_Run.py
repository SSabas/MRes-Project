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
import sys

# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')

from A_Data_Import import *
from B_Moment_Estimation import *
from C_Simulation import *
from D_CPLEX_Solver import *
from E_Evaluation import *

# -------------------- VARIABLES ----------------------------------------------------- #

input_file = 'moment_estimations'
output_file = 'scenario_tree'
simulations = 100000
nr_scenarios = 4
branching = (2, 2, 8, 8)
instruments_NYSE = ['KO', 'MSFT', 'IBM', 'AXP', 'PG', 'DIS', 'INTC', 'FDX', 'ADM', 'MAT']
instruments_FTSE = ['HSBC', 'VOD', 'BP', 'GSK', 'AZN', 'RIO', 'BG', 'TSCO', 'BT', 'PRU']
instruments = instruments_FTSE #+ instruments_FTSE
start_date = '2005-01-01'
end_date = '2014-01-01'
source = 'morningstar'
price_point = 'Close'
to_plot = 'yes'
to_save = 'yes'
initial_portfolio = np.repeat(1/len(instruments), len(instruments)) # Equally weighted portfolio
beta = 0.99
cost_to_sell = 0.01
cost_to_buy = 0.01
initial_wealth = 1
return_target = 1.01
solver = 'gurobi'
frequency = 'weekly'
look_back_period = 50
input_file = 'moment_estimation'
benchmark = 'yes'
periods_to_forecast = 5
folder = 'portfolio_optimisation_%s_weeks' %(periods_to_forecast)

# Bounds for optimisation
sell_bounds = [[0.0], [0.2]]
buy_bounds = [[0.0], [0.2]]
weight_bounds = [[0.0], [1.0]]

# Improves readability (extends)
pd.set_option('display.max_columns', 10)

# Specify testing parameters
returns = np.linspace(1.02, 1.10, 20)

# -------------------- EXECUTING C++ CODE THROUGH PYTHON ----------------------------- #

# Get the data
stock_data = import_stock_data_api(instruments=instruments, data_source=source,
                                   start_date=start_date, end_date=end_date, price_point=price_point,
                                   to_plot=to_plot, to_save=to_save, from_file='no', folder=folder,
                                   frequency=frequency)  # Takes c. 20 secs to query

# Test exponential fit for a single stock
# residuals, parameters = exponential_growth(stock_data['VOD'], to_plot=to_plot)

# Test moment-calculation
# means, variances = calculate_moments(stock_data)

# Get the moments and save to format for the simulator
# put_to_cpp_layout(folder, input_file, stock_data, branching=branching)

# Run the simulation
# clean_cluster() # In case first run
# compile_cluster() # Gives some errors, but still works
# run_cluster(input_file, output_file, folder, samples=simulations, scenario_trees=nr_scenarios)

# Read the output
# scenarios_dict = read_cluster_output(output_file, folder, scenario_trees=nr_scenarios, asset_names=instruments)

# Get the final cumulative probabilities
# scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

# Plot the simulation output for sense-checking
# output = {}
# output_cum = {}
# for i in instruments:
#     print(i)
#     output[i], output_cum[i] = plot_cluster_output(stock_data, i, scenarios_dict['1'], branching, to_plot='yes')

# Create an efficient frontier
# ef_wcvars = efficient_frontier(scenarios_dict, returns, instruments, branching, initial_portfolio,
#                        sell_bounds, buy_bounds, weight_bounds, cost_to_buy=cost_to_buy, cost_to_sell=cost_to_sell,
#                        beta=beta, initial_wealth=initial_wealth, to_plot=to_plot, folder=folder, solver=solver,
#                        to_save=to_save)

# Calculate the optimised portfolio
optimised_returns, benchmark_returns = portfolio_optimisation(stock_data, look_back_period, start_date, end_date,
                                                              folder=folder, periods_to_forecast=periods_to_forecast,
                                                              input_file=input_file, frequency=frequency,
                                                              benchmark=benchmark, to_plot=to_plot, to_save=to_save,
                                                              branching=branching, simulations=simulations,
                                                              initial_portfolio=initial_portfolio,
                                                              nr_scenarios=nr_scenarios, return_target=return_target,
                                                              sell_bounds=sell_bounds, buy_bounds=buy_bounds,
                                                              weight_bounds=weight_bounds, cost_to_buy=cost_to_buy,
                                                              cost_to_sell=cost_to_sell, beta=beta,
                                                              initial_wealth=initial_wealth, solver=solver)

# Save the workspace variables to file


