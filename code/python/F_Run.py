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
from operator import mul
from functools import reduce


# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')

from A_Data_Import import *
from B_Moment_Estimation import *
from C_Simulation import *
from D_CPLEX_Solver import *


# -------------------- VARIABLES ----------------------------------------------------- #

folder = 'test_run'
input_file = 'moment_estimations'
output_file = 'scenario_tree'
simulations = 40000
nr_scenarios = 10
branching = (10, 10, 2)
instruments_NYSE = ['KO', 'MSFT', 'IBM', 'AXP', 'PG', 'DIS', 'INTC', 'FDX', 'ADM', 'MAT']
instruments_FTSE = ['HSBC', 'VOD', 'BP', 'GSK', 'AZN', 'RIO', 'BG', 'TSCO', 'AAL', 'PRU']
instruments = instruments_FTSE #+ instruments_FTSE
start_date = '2015-01-01'
end_date = '2017-01-01'
source = 'morningstar'
price_point = 'Close'
to_plot = 'yes'
initial_portfolio = np.repeat(1/len(instruments), len(instruments)) # Equally weighted portfolio
beta = 0.99
cost_to_sell = 0.01
cost_to_buy = 0.01
initial_wealth= 1
return_target = 1.02
solver = 'gurobi'

# Bounds for optimisation
sell_bounds = [[0.0], [0.2]]
buy_bounds = [[0.0], [0.2]]
weight_bounds = [[0.0], [1.0]]

# Improves readability (extends)
pd.set_option('display.max_columns', 10)

# Specify testing parameters
returns = np.linspace(1.02, 1.08, 20)

# -------------------- EXECUTING C++ CODE THROUGH PYTHON ----------------------------- #

# Get the data
stock_data = import_stock_data_api(instruments=instruments, data_source= source,
                                   start_date=start_date, end_date=end_date, price_point=price_point,
                                   to_plot=to_plot, to_save='yes', from_file='no')  # Takes c. 20 secs to query

# Test exponential fit for a single stock
# residuals, parameters = exponential_growth(stock_data['VOD'], to_plot=to_plot)

# Test moment-calculation
# means, variances = calculate_moments(stock_data)

# Get the moments and save to format for the simulator
put_to_cpp_layout(folder, input_file, stock_data, branching=branching)

# Run the simulation
# clean_cluster() # In case first run
# compile_cluster() # Gives some errors, but still works
run_cluster(input_file, output_file, folder, samples=simulations, scenario_trees=nr_scenarios)

# Read the output
scenarios_dict = read_cluster_output(output_file, folder, scenario_trees=nr_scenarios, asset_names=instruments)

# Get the final cumulative probabilities
scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

# Plot the simulation output for sense-checking
output = {}
output_cum = {}
for i in instruments:
    print(i)
    output[i], output_cum[i] = plot_cluster_output(stock_data, i, scenarios_dict['1'], branching, to_plot='yes')



answer, w_0_weights, variables = robust_portfolio_optimisation(scenarios_dict, instruments, branching, initial_portfolio,
                                  sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                  beta=0.99, initial_wealth=1, return_target=1.06, to_save='yes', folder=folder)

answer, w_0_weights, variables = robust_portfolio_optimisation(scenarios_dict, instruments, branching, initial_portfolio,
                                  sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                  beta=0.95, initial_wealth=1, return_target=1.06, to_save='no', folder='',
                                  solver='gurobi', wcvar_minimizer='yes')

    wcvars.append(answer.solution.get_objective_value())

return_targets = np.linspace(1, 1.08, 20)
wcvars = []
wcvars_1 = []
wcvars_2 = []
wcvars_3 = []
wcvars_4 = []


dict_1 = { '1': scenarios_dict['1']}
dict_2 = { '1': scenarios_dict['2']}
dict_3 = { '1': scenarios_dict['3']}
dict_4 = { '1': scenarios_dict['4']}


for i in return_targets:
    print(i)
    answer, w_0_weights, variables = robust_portfolio_optimisation(scenarios_dict, instruments, branching, initial_portfolio,
                                      sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                      beta=0.99, initial_wealth=1, return_target=i, to_save='no', folder=folder)
    wcvars.append(answer.solution.get_objective_value())

    answer, w_0_weights, variables = robust_portfolio_optimisation(dict_1, instruments, branching, initial_portfolio,
                                      sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                      beta=0.99, initial_wealth=1, return_target=i, to_save='no', folder=folder)
    wcvars_1.append(answer.solution.get_objective_value())

    answer, w_0_weights, variables = robust_portfolio_optimisation(dict_2, instruments, branching, initial_portfolio,
                                      sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                      beta=0.99, initial_wealth=1, return_target=i, to_save='no', folder=folder)
    wcvars_2.append(answer.solution.get_objective_value())

    answer, w_0_weights, variables = robust_portfolio_optimisation(dict_3, instruments, branching, initial_portfolio,
                                      sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                      beta=0.99, initial_wealth=1, return_target=i, to_save='no', folder=folder)
    wcvars_3.append(answer.solution.get_objective_value())

    answer, w_0_weights, variables = robust_portfolio_optimisation(dict_4, instruments, branching, initial_portfolio,
                                      sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                                      beta=0.99, initial_wealth=1, return_target=i, to_save='no', folder=folder)
    wcvars_4.append(answer.solution.get_objective_value())

plt.plot(wcvars, return_targets, label='All')
plt.plot(wcvars_1, return_targets, label='1')
plt.plot(wcvars_2, return_targets, label='2')
plt.plot(wcvars_3, return_targets, label='3')
plt.plot(wcvars_4, return_targets, label='4')
plt.legend()

plt.savefig('name.png')
plt.show()



import matplotlib
import matplotlib.pyplot
print(matplotlib.backends.backend)