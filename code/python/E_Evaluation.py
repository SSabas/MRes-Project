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
  14/06/2018
"""

# ------------------------------ IMPORT LIBRARIES --------------------------------- #

import matplotlib.pyplot as plt
import os
import json
import sys
from datetime import datetime, timedelta
import math

# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')
from D_CPLEX_Solver import *


# ----------------------------- EVALUATE THE OPTIMISER ---------------------------- #

# Creates an efficient Return-CVaR frontier
def efficient_frontier(scenarios_dict, returns, instruments, branching, initial_portfolio,
                       sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                       beta=0.99, initial_wealth=1, to_plot='yes', folder='', solver='qurobi',
                       to_save='yes'):

    # Create dictionary of dictionaries for testing and also the results dictionary
    test_dict = {}
    results = {}
    for tree in scenarios_dict:
        # print(tree)
        test_dict[tree] = {tree: scenarios_dict[tree]}
        results[tree] = []

    # Add also the combined output
    results['All'] = []

    # Iterate over the returns
    print('Running the optimisation.')
    for i,j in zip(returns, range(1, len(returns)+1)):
        print('Cycle number %s (out of %s).' %(j, len(returns)))

        # Firstly run the full optimiser
        print('Optimising with full set of trees (%s trees in total) in cycle number %s (out of %s).'
              %(len(list(scenarios_dict.keys())), j, len(returns)))
        answer, w_0_weights, variables = robust_portfolio_optimisation(scenarios_dict, instruments, branching,
                                                                       initial_portfolio, sell_bounds, buy_bounds,
                                                                       weight_bounds, cost_to_buy=cost_to_buy,
                                                                       cost_to_sell=cost_to_sell, beta=beta,
                                                                       initial_wealth=initial_wealth, return_target=i,
                                                                       folder=folder, solver = solver)
        results['All'].append(answer)#.solution.get_objective_value())

        # Run each tree separately
        for tree in test_dict:
            print('Optimising with only single tree (tree number %s) in cycle number %s (out of %s).'
                  %(tree, j, len(returns)))
            answer, w_0_weights, variables = robust_portfolio_optimisation(test_dict[tree], instruments, branching,
                                                                           initial_portfolio,
                                                                           sell_bounds, buy_bounds, weight_bounds,
                                                                           cost_to_buy=cost_to_buy,
                                                                           cost_to_sell=cost_to_sell,
                                                                           beta=beta, initial_wealth=initial_wealth,
                                                                           return_target=i, folder=folder,
                                                                           solver= solver)
            results[tree].append(answer)#.solution.get_objective_value())

    if to_plot == 'yes':

        print('Running the optimisation.')
        for i in results:
            # print(i)
            # if np.max(results[i]) > 5:
            #     continue
            #
            # else:
            if i == "All":
                plt.plot(results[i], returns, label='Min-max with %s trees' %len(test_dict), linestyle ='--')
            else:
                plt.plot(results[i], returns, label ='Tree %s' %i)

            plt.legend()
            plt.title('Mean-Robust CVaR Efficient Frontiers')
            plt.ylabel('Return')
            plt.xlabel('Conditional Value-at-Risk')

    if to_save == 'yes':

        # Save the plot
        plt.savefig(os.getcwd() + '/results/'+ folder + '/mean_robust_cvar_efficient_frontier.pdf')

        # Save the data dictionary
        json_file = json.dumps(results)
        f = open(os.getcwd() + '/results/'+ folder + "/efficient_frontier_dict.json", "w")
        f.write(json_file)
        f.close()

    return results


# Portfolio optimisation

def portfolio_optimisation(stock_data, look_back_period, start_date, end_date, days_to_forecast=None,
                           frequency='daily', to_save='yes', folder=None, benchmark='yes'):

    # If days_to_forecast is not present, use start and end date to get how many days to forecast
    if days_to_forecast is None:

        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        days_to_forecast = (end_date - start_date + timedelta(days=1)).days # to count the first day  too

    # If using weekly data, change the days_to_forecast to weeks
    if frequency == 'weekly':
        days_to_forecast = math.floor(days_to_forecast/7)

    # Check whether there is enough back-testing/fitting data to accommodate the look_back_period an days_to_forecast

    length_stock_data = len(stock_data)

    if length_stock_data < (days_to_forecast + look_back_period):

        raise ValueError('Dataset is too short. The dataset has %s observations, but the specified parameters '
                         'require %s (look-back period of %s and forecast period of %s).' %(length_stock_data,
                                                                                 days_to_forecast + look_back_period,
                                                                                 look_back_period, days_to_forecast))
    # Iterate over dataset
    for i in range(days_to_forecast):
        print(i)

        # Get back-fitting data
        end_date_back_fitting = end_date - timedelta(days=1)
        start_date_back_fitting = end_date_back_fitting - timedelta(days=look_back_period)

    stock_data[start_date_back_fitting:end_date_back_fitting]