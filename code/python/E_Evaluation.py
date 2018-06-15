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

# Add the python scripts folder to system path
sys.path.insert(0, os.getcwd() + '/code/python')
from D_CPLEX_Solver import *


# ----------------------------- EVALUATE THE OPTIMISER ---------------------------- #

def efficient_frontier(scenarios_dict, returns, instruments, branching, initial_portfolio,
                       sell_bounds, buy_bounds, weight_bounds, cost_to_buy=0.01, cost_to_sell=0.01,
                       beta=0.99, initial_wealth=1, to_plot='yes', folder=folder, solver='qurobi',
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
            print(i)
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
