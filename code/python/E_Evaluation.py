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
import numpy as np
import pandas as pd

# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')

from A_Data_Import import *
from B_Moment_Estimation import *
from C_Simulation import *
from D_CPLEX_Solver import *


# ----------------------------- EVALUATE THE OPTIMISER ---------------------------- #

# Creates an efficient Return-CVaR frontier
def efficient_frontier(stock_data, branching, initial_portfolio, simulations=100000, return_points=5,
                       nr_scenarios=4, sell_bounds=None, buy_bounds=None, weight_bounds=None, cost_to_buy=0.01,
                       cost_to_sell=0.01, beta=0.99, initial_wealth=1, to_plot='yes', folder='', solver='qurobi',
                       to_save='yes', input_file='moment_estimation'):


    # Infer some metadata from inputs
    instruments = list(stock_data.columns)

    # Define output file
    output_file = 'scenario_file'

    # Get the moments and save to format for the simulator
    put_to_cpp_layout(folder, input_file, stock_data, branching=branching)

    # Run the simulation
    run_cluster(input_file, output_file, folder=folder, samples=simulations, scenario_trees=nr_scenarios)

    # Read the output
    scenarios_dict = read_cluster_output(output_file, folder, scenario_trees=nr_scenarios,
                                         asset_names=instruments)

    # Get the final cumulative probabilities
    scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

    # Get minimum return
    min_return = robust_portfolio_optimisation(scenarios_dict, instruments, branching, initial_portfolio, sell_bounds,
                                               buy_bounds, weight_bounds, cost_to_buy=cost_to_buy,
                                               cost_to_sell=cost_to_sell,
                                               beta=beta, initial_wealth=initial_wealth, to_save='no', folder=folder,
                                               solver='cplex', wcvar_minimizer='yes')  # Use CPLEX for minimization

    # Get maximum return
    max_return = return_maximisation(scenarios_dict, instruments, branching, initial_portfolio, sell_bounds, buy_bounds,
                                     weight_bounds, cost_to_buy=cost_to_buy, cost_to_sell=cost_to_sell, beta=beta,
                                     initial_wealth=initial_wealth, folder=folder, solver='cplex')

    # Construct return sequence
    returns = np.linspace(min_return['return'], max(max_return.values()), return_points)

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
                                                                       folder=folder, solver=solver)
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
        plt.figure(figsize=(9, 6))
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
            plt.tight_layout()

    if to_save == 'yes':

        # Save the plot
        plt.savefig(os.getcwd() + '/results/'+ folder + '/mean_robust_cvar_efficient_frontier.pdf')

        # Save the data dictionary
        json_file = json.dumps(results)
        f = open(os.getcwd() + '/results/'+ folder + "/efficient_frontier_dict.json", "w")
        f.write(json_file)
        f.close()

    return results


def efficient_portfolio_variance_testing(stock_data, branching, initial_portfolio, simulations=100000, return_points=5,
                                         nr_scenarios=4, sell_bounds=None, buy_bounds=None, weight_bounds=None,
                                         cost_to_buy=0.01, cost_to_sell=0.01, beta=0.99, initial_wealth=1,
                                         to_plot='yes', folder='', solver='qurobi', to_save='yes',
                                         input_file='moment_estimation', min_return=None, max_return=None, samples=10,
                                         min_max_adjustment=None):

    # Infer some metadata from inputs
    instruments = list(stock_data.columns)

    # Set initial portfolio if it is not specified
    if initial_portfolio is None:
        initial_portfolio = np.repeat(1 / len(stock_data.columns), len(stock_data.columns))

    if min_return is None and max_return is None:
        print('Infering the minimum and maximum returns from the data')

        # Define output file
        output_file = 'scenario_file'

        # Define the folder
        folder_min_max = folder + '/min_max'

        # Get the moments and save to format for the simulator
        put_to_cpp_layout(folder_min_max, input_file, stock_data, branching=branching)

        # Run the simulation
        run_cluster(input_file, output_file, folder=folder_min_max, samples=simulations, scenario_trees=nr_scenarios)

        # Read the output
        scenarios_dict = read_cluster_output(output_file, folder_min_max, scenario_trees=nr_scenarios,
                                             asset_names=instruments)

        # Get the final cumulative probabilities
        scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

        # Get minimum return
        if min_return is None:
            folder_min = folder_min_max + "/min"
            min_return = robust_portfolio_optimisation(scenarios_dict, instruments, branching, initial_portfolio,
                                                       sell_bounds, buy_bounds, weight_bounds, cost_to_buy=cost_to_buy,
                                                       cost_to_sell=cost_to_sell,
                                                       beta=beta, initial_wealth=initial_wealth, to_save='yes',
                                                       folder=folder_min, solver='cplex',
                                                       wcvar_minimizer='yes')  # Use CPLEX for minimization
            min_return = min_return['return']

        # Get maximum return
        if max_return is None:
            folder_max = folder_min_max + "/max"
            max_return = return_maximisation(scenarios_dict, instruments, branching, initial_portfolio, sell_bounds,
                                             buy_bounds, weight_bounds, cost_to_buy=cost_to_buy,
                                             cost_to_sell=cost_to_sell, beta=beta, to_save='yes',
                                             initial_wealth=initial_wealth, folder=folder_max, solver='cplex')
            max_return = max(max_return.values())

    # Construct return sequence
    if min_max_adjustment is not None:

        # Calculate new min_return boundary based on the confidence level
        adjustment = (max_return - min_return)*((1-min_max_adjustment)/2)
        min_return_adj = min_return + adjustment
        max_return_adj = max_return - adjustment

    returns = np.linspace(min_return, max_return, return_points)
    returns = np.round(returns, 3)

    # Create dictionary of dictionaries to store the results
    results = {}
    for i in returns:
        results[i] = []

    # Iterate over the returns
    print('Running the optimisation.')
    for i, j in zip(returns, range(1, len(returns) + 1)):
        print('Cycle number %s (out of %s).' % (j, len(returns)))

        # Specify folder
        folder_returns = folder + '/return_iteration_%s' %j

        for sample in range(1, samples+1):
            print('Sample number %s (out of %s) in return cycle %s (out of %s).' % (sample, samples, j, len(returns)))

            # Specify the folder
            folder_sample = folder_returns + '/sample_%s' %sample

            # Get the moments and save to format for the simulator
            put_to_cpp_layout(folder_sample, input_file, stock_data, branching=branching)

            # Run the simulation
            run_cluster(input_file, output_file, folder=folder_sample, samples=simulations, scenario_trees=nr_scenarios)

            # Read the output
            scenarios_dict = read_cluster_output(output_file, folder_sample, scenario_trees=nr_scenarios,
                                                 asset_names=instruments)

            # Get the final cumulative probabilities
            scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

            # Run the optimisation
            answer, w_0_weights, variables = robust_portfolio_optimisation(scenarios_dict, instruments, branching,
                                                                           initial_portfolio, sell_bounds, buy_bounds,
                                                                           weight_bounds, cost_to_buy=cost_to_buy,
                                                                           cost_to_sell=cost_to_sell, beta=beta,
                                                                           initial_wealth=initial_wealth, return_target=i,
                                                                           folder=folder_sample, solver=solver)

            # Store the results
            results[i].append(answer)

    # Retrieve the result and put to pandas dataframe
    results_df = pd.DataFrame.from_dict(results)

    # Plot results
    if to_plot == 'yes':

        # Boxplot
        plt.figure(figsize=(9, 6))
        results_df.boxplot()
        plt.title('Boxplots of CVaR Optimisation (%s Samples per Return Specification)' %samples)
        plt.xlabel('Return')
        plt.ylabel('CVaR')
        plt.tight_layout()

        if to_save == 'yes':
            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/boxplot_returns_vs_cvar.pdf')

        # Line plot
        means = np.mean(results_df, axis=0)
        lower_95 = results_df.quantile(0.025)
        upper_95 = results_df.quantile(0.975)
        lower_80 = results_df.quantile(0.1)
        upper_80 = results_df.quantile(0.90)

        # Plot the figure
        plt.figure(figsize=(9, 6))
        plt.plot(means.index, means, linewidth=1.2, markersize=6, color='green', label='Mean CVaR')
        plt.fill_between(means.index, upper_80, lower_80,
                         color='green', alpha=.25, lw=0, label='80% Confidence Interval')
        plt.fill_between(means.index, upper_95, lower_95,
                         color='green', alpha=.1, lw=0, label='95% Confidence Interval')
        plt.title('Lineplot of CVaR Optimisation (%s Samples per Return Specification)' %samples)
        plt.xlabel('Return')
        plt.ylabel('CVaR')
        plt.legend()
        plt.tight_layout()

        if to_save == 'yes':
            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/lineplot_returns_vs_cvar.pdf')

        # Plot also standard deviations
        std = np.std(results_df, axis=0)
        plt.figure(figsize=(9, 6))
        plt.plot(std.index, std, linewidth=1.2, markersize=6, color='green')
        plt.title('Standard Deviation of CVaR Optimisation (%s Samples per Return Specification)' %samples)
        plt.xlabel('Return')
        plt.ylabel('Standard Deviation of CVaR')
        plt.tight_layout()

        if to_save == 'yes':
            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/std_returns_vs_cvar.pdf')

    # Save results
    if to_save == 'yes':
        # Save the data dictionary
        results_df.to_csv(os.getcwd() + '/results/' + folder + '/returns_vs_cvar_data.csv')

    return results


def compare_efficient_frontier_variance_tests(*args):

    # Unpack the arguments, that should be folders/paths to the outputs of variance tests

    # Define plot parameters
    plt.figure(figsize=(9, 6))
    plt.xlabel('Return')
    plt.ylabel('CVaR')
    plt.tight_layout()
    colors = ['forestgreen', 'royalblue', 'firebrick', 'orange']

    for i in range(0, len(args)):
        print(i, args[i])

        # Import data
        results_df = pd.read_csv(os.getcwd() + '/results/' + args[i] + '/returns_vs_cvar_data.csv',
                                 index_col=0)

        means = np.mean(results_df, axis=0)
        lower_95 = results_df.quantile(0.025)
        upper_95 = results_df.quantile(0.975)
        label_variables = args[i].split('_')
        label = '%s Branching with %s Scenario Tree(s)' %("-".join(label_variables[8]), label_variables[10])

        # Plot
        plt.plot(means.index.astype(float), means, linewidth=1.2, markersize=6, color=colors[i], label=label)
        plt.fill_between(means.index.astype(float), upper_95, lower_95,
                         color=colors[i], alpha=.25, lw=0)
        plt.title('CVaR Optimisation Comparison (%s Samples per Return Specification)' % samples)

    plt.legend()

    # Save file
    plt.savefig(os.getcwd() + '/results/cvar_variance_comparison.pdf')


# Portfolio optimisation
def portfolio_optimisation(stock_data, look_back_period, folder=None,
                           periods_to_forecast=None, input_file='moment_estimation',
                           frequency='weekly', benchmark='yes', to_plot='yes', to_save='yes',
                           branching=(2, 2, 8, 8), simulations=100000,
                           initial_portfolio=None,
                           nr_scenarios=256, return_target=1.05, sell_bounds=None, buy_bounds=None,
                           weight_bounds=None, cost_to_buy=0.01, cost_to_sell=0.01, beta=0.99, initial_wealth=1,
                           solver='gurobi'):

    # Infer some metadata from inputs
    instruments = list(stock_data.columns)

    # Iterator to loop
    count_iterator = timedelta(days=1)

    if initial_portfolio is None:
        initial_portfolio = np.repeat(1 / len(stock_data.columns), len(stock_data.columns))

    if to_save != 'yes':
        folder = None

    # If days_to_forecast is not present, use start and end date to get how many days to forecast
    # if periods_to_forecast is None:
    #
    #     # start_date = datetime.strptime(start_date, '%Y-%m-%d')
    #     # end_date = datetime.strptime(end_date, '%Y-%m-%d')
    #     periods_to_forecast = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    #                            + timedelta(days=1)).days # to count the first day  too

    # If using weekly data, change the days_to_forecast to weeks
    if frequency == 'weekly':
        count_iterator = timedelta(days=7)

    # Check whether there is enough back-testing/fitting data to accommodate the look_back_period an days_to_forecast

    length_stock_data = len(stock_data)

    if length_stock_data < (periods_to_forecast + look_back_period):

        raise ValueError('Dataset is too short. The dataset has %s observations, but the specified parameters '
                         'require %s (look-back period of %s and forecast period of %s).' %(length_stock_data,
                                                                                            periods_to_forecast
                                                                                            + look_back_period,
                                                                                            look_back_period,
                                                                                            periods_to_forecast))

    # Create returns dataset
    returns_data = stock_data.iloc[-periods_to_forecast:]
    returns_data = returns_data.pct_change() + 1

    # Create benchmark dataset
    benchmark_returns = stock_data.iloc[-periods_to_forecast:].pct_change()+1
    benchmark_returns.iloc[0, :] = 1/len(instruments)
    benchmark_returns = benchmark_returns.cumprod().sum(1)

    # Create an empty dataset for optimised returns
    optimised_returns = stock_data.copy()[-periods_to_forecast:]
    optimised_returns.iloc[:, ] = np.nan
    optimised_returns.iloc[0, ] = initial_portfolio

    # Set dates to back-testing
    if frequency == 'daily':
        end_date_back_fitting = stock_data.index[-1] - timedelta(days=periods_to_forecast)
        start_date_back_fitting = end_date_back_fitting - timedelta(days=look_back_period)
        # stock_data.index[-1] - timedelta(days=look_back_period- 1)

    else:
        end_date_back_fitting = stock_data.index[-1] - timedelta(days=periods_to_forecast*7)
        start_date_back_fitting = end_date_back_fitting - timedelta(days=look_back_period*7)
        # stock_data.index[-1] - timedelta(days=look_back_period- 1)


    # Define output file
    output_file = 'scenario_file'

    # Iterate over dataset
    for i in range(periods_to_forecast-1):
        print('Optimisation iteration number %s (out of %s).' %(i, periods_to_forecast-1))

        # Iterate over scenario folders
        scenario_folder = folder + '/period_%s' %(i+1)

        # Get back-fitting data
        bs_data = stock_data[start_date_back_fitting:end_date_back_fitting]

        # Run the simulation
        # Get the moments and save to format for the simulator
        put_to_cpp_layout(scenario_folder, input_file, bs_data, branching=branching)

        # Run the simulation
        # clean_cluster() # In case first run
        # compile_cluster() # Gives some errors, but still works
        run_cluster(input_file, output_file, folder=scenario_folder, samples=simulations, scenario_trees=nr_scenarios)

        # Read the output
        scenarios_dict = read_cluster_output(output_file, scenario_folder, scenario_trees=nr_scenarios, asset_names=instruments)

        # Get the final cumulative probabilities
        scenarios_dict = add_cumulative_probabilities(scenarios_dict, branching)

        # Run the optimisation
        wcvar, variables, w_0_weights = robust_portfolio_optimisation(scenarios_dict, instruments, branching,
                                                                      initial_portfolio, sell_bounds, buy_bounds,
                                                                      weight_bounds, cost_to_buy=cost_to_buy,
                                                                      cost_to_sell=cost_to_sell, beta=beta,
                                                                      initial_wealth=initial_wealth,
                                                                      return_target=return_target,
                                                                      to_save='yes', folder=scenario_folder,
                                                                      solver=solver)

        # Add the weights to the optimised returns dataset
        reweighted_optimal_portfolio = w_0_weights/np.sum(w_0_weights) # Make the portfolio to 1
        previous_weights = optimised_returns.iloc[i, :]

        # Deduct the transaction costs from periods t = 2, ..., T
        if i != 0:

            # Get all transactions
            transactions = reweighted_optimal_portfolio * np.sum(optimised_returns.iloc[i, :]) - previous_weights

            # Cost of sells
            sells = transactions < 0
            sell_costs = np.sum(np.abs(transactions[sells]) * cost_to_sell)

            # Cost of buys
            buys = transactions > 0
            buy_costs = np.sum(np.abs(transactions[buys]) * cost_to_buy)

            # Total cost
            total_cost = sell_costs + buy_costs

            # Transaction cost adjusted portfolio
            optimised_returns.iloc[i, :] = reweighted_optimal_portfolio*(np.sum(optimised_returns.iloc[i, :]) - total_cost) # Reweight over the current value of portfolio minus transaction costs
            # np.sum(np.abs(reweighted_optimal_portfolio * np.sum(optimised_returns.iloc[i, :]) - previous_period_weights))

        else:
            optimised_returns.iloc[i, :] = reweighted_optimal_portfolio*np.sum(optimised_returns.iloc[i, :]) # Reweight over the current value of portfolio

        # Calculate the portfolio balance for next period
        optimised_returns.iloc[i+1, :] = optimised_returns.iloc[i, :] * returns_data.iloc[i+1, :]

        # Reset initial portfolio value to current portfolio (normalised to 1 for convenience)
        initial_portfolio = np.array(optimised_returns.iloc[i+1, :] / np.sum(optimised_returns.iloc[i+1, :]))

        # Move dates forward
        start_date_back_fitting = start_date_back_fitting + count_iterator
        end_date_back_fitting = end_date_back_fitting + count_iterator

    # Create output dictionary
    output = {}
    output['optimised_portfolio'] = optimised_returns

    # Calculate the realised CVaR of the optimised portfolio
    aggregated_portfolio_returns = optimised_returns.sum(1).pct_change()
    portfolio_var = aggregated_portfolio_returns.quantile(q=1-beta, interpolation='lower')
    portfolio_cvar = -np.mean(aggregated_portfolio_returns[aggregated_portfolio_returns<=portfolio_var])
    output['portfolio_cvar'] = portfolio_cvar

    # Same for benchmark if present
    if benchmark == 'yes':

        aggregated_benchmark_returns = benchmark_returns.pct_change()
        benchmark_var = aggregated_benchmark_returns.quantile(q=1 - beta, interpolation='lower')
        benchmark_cvar = -np.mean(aggregated_benchmark_returns[aggregated_benchmark_returns <= benchmark_var])
        output['benchmark_cvar'] = benchmark_cvar
        output['benchmark_portfolio'] = benchmark_returns

    if to_plot == 'yes':

        plt.figure(figsize=(9, 6))
        # Calculate the cumulative returns
        data_to_plot = optimised_returns.sum(1)
        plt.plot(data_to_plot, label='Min-max Optimised Portfolio (CVaR = %.3f)' %output['portfolio_cvar'])

        if benchmark == 'yes':
            plt.plot(benchmark_returns, label='Equally Weighted Portfolio (CVaR = %.3f)' %output['benchmark_cvar'])

        plt.legend()
        plt.title('Performance Comparison')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Date')
        plt.tight_layout()

        if to_save == 'yes':

            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/optimised_portfolio_analysis.pdf')

        # Create plot of portfolio weights
        optimised_returns.plot.area(figsize=(9, 6))
        plt.title('Min-Max CVaR Optimised Portfolio Weights')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Date')
        plt.tight_layout()

        if to_save == 'yes':

            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/optimised_portfolio_weights.pdf')

    if to_save == 'yes':

        # Save the data
        optimised_returns.to_csv(os.getcwd() + '/results/' + folder + '/optimised_portfolio_data.csv')

    return output


# Function to test the variance of the portfolio optimisation framework
def portfolio_optimisation_variance_testing(stock_data, look_back_period, folder=None,
                                            periods_to_forecast=None, input_file='moment_estimation',
                                            frequency='weekly', benchmark='yes', to_plot='yes', to_save='yes',
                                            branching=(2, 2, 8, 8), simulations=100000, initial_portfolio=None,
                                            nr_scenarios=256, return_target=1.05, sell_bounds=None, buy_bounds=None,
                                            weight_bounds=None, cost_to_buy=0.01, cost_to_sell=0.01, beta=0.99,
                                            initial_wealth=1, solver='gurobi', iterations=4):

    # Create an output dictionary
    output_dict = {}

    # Run the iterations
    for i in range(1, iterations+1):
        folder_portfolio = folder + '/test_%s' %i
        print('Iteration number %s.' %i)
        output_dict[str(i)] = portfolio_optimisation(stock_data, look_back_period, folder=folder_portfolio,
                                                     periods_to_forecast=periods_to_forecast, input_file=input_file,
                                                     frequency=frequency, benchmark=benchmark, to_plot='no',
                                                     to_save='yes', branching=branching, simulations=simulations,
                                                     initial_portfolio=initial_portfolio, nr_scenarios=nr_scenarios,
                                                     return_target=return_target, sell_bounds=sell_bounds,
                                                     buy_bounds=buy_bounds, weight_bounds=weight_bounds,
                                                     cost_to_buy=cost_to_buy, cost_to_sell=cost_to_sell, beta=beta,
                                                     initial_wealth=initial_wealth, solver=solver)

    # Plot the results and save the file
    if to_plot == 'yes':

        plt.figure(figsize=(9, 6))

        for j in output_dict:

            # Get the data
            data_to_plot = output_dict[j]

            # Calculate the cumulative returns
            optimised_portfolio = data_to_plot['optimised_portfolio'].sum(1)

            # Plot
            plt.plot(optimised_portfolio, label=('Min-max Optimised Portfolio (CVaR = %.3f) (Sample %s)' % (data_to_plot['portfolio_cvar'], j)))

        if benchmark == 'yes':

            # Get the data
            data_to_plot = output_dict[j]

            # Calculate the cumulative returns
            benchmark_portfolio = data_to_plot['benchmark_portfolio']

            plt.plot(benchmark_portfolio, label='Equally Weighted Portfolio (CVaR = %.3f)' %
                                                data_to_plot['benchmark_cvar'], linestyle='--')

        plt.legend()
        plt.title('Performance Comparison')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Date')
        plt.tight_layout()

        if to_save == 'yes':

            # Save the plot
            plt.savefig(os.getcwd() + '/results/' + folder + '/optimised_portfolio_analysis_all.pdf')

    return output_dict

