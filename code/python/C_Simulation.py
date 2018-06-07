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
from functools import reduce
from matplotlib.ticker import MaxNLocator

# -------------------- PYTHON WRAPPER FOR CLUSTER MODULE ----------------------------- #


def clean_cluster():

    # Specify path
    path = os.getcwd() + "/code/cpp/cluster2"

    # Clean the compiled code
    clean = subprocess.Popen(["make", "clean"], cwd=path)


def compile_cluster():

    # Specify path to makefile
    path = os.getcwd() + "/code/cpp/cluster2"

    # Extract the content
    makefile = open(path + "/Makefile", "r")
    lines = makefile.readlines()
    makefile.close()

    # Change the directory
    lines[-3] = '\tcp ' + os.getcwd() + '/code/cpp/cluster2/' + '\n'

    # Save output
    makefile = open(path + "/Makefile", 'w')

    for line in lines:
        makefile.write("%s" % line)

    makefile.close()

    # Compile/install the library
    subprocess.call(["make", "install"], shell=True, cwd=path)


def run_cluster(input_file, output_file, samples=10000):

    # Define the inputs
    path = os.getcwd() + '/code/cpp/cluster2/'
    inputs = '-f ' + input_file
    outputs = ' -o ' + output_file
    simulations = ' -n ' + str(samples)
    combined = inputs + outputs + simulations

    # Run te process
    subprocess.call('./cluster2 ' + combined, shell=True, cwd=path)


def read_cluster_output(output_file, asset_names=('KO', 'F', 'IBM', 'AXP', 'PG')):

    # Complete the path
    path = os.getcwd() + '/code/cpp/cluster2/' + output_file

    # Extract the content
    makefile = open(path, "r")
    lines = makefile.readlines()
    makefile.close()

    # Get how many assets are
    assets = int(lines[1][7])  # Hence would ignore 4 + (N*N-N)/2 + N + 1 lines in the beginning

    # Get the scenarios
    begin = int(4 + (assets*assets-assets)/2 + assets + 1)
    end = len(lines) - 6
    scenarios = lines[begin:end]

    # Extract the scenarios and format the output
    scenarios_list = list([line.replace('\t', ' ').replace(' \n', '').split(' ') for line in scenarios])
    scenarios_df = pd.DataFrame(scenarios_list)
    scenarios_df.columns = np.array(['node', 'probability', *asset_names])

    return scenarios_df


def format_cluster_output(data, instrument, scenarios, branching, to_plot='yes'):

    # Define dataframe size
    forecast_steps = len(branching)
    final_scenarios = reduce(lambda x, y: x*y, branching)

    # Define the dataframe to be populated
    output = pd.DataFrame(np.random.randint(low=1, high=10, size=(final_scenarios, forecast_steps+1)))
    output[0] = list(scenarios[-final_scenarios:]['node'])
    output.index = scenarios[-final_scenarios:]['node']

    # Change the column names
    # names = list(output.columns.values)
    # names[0] = 'node'
    # output.columns = names
    output['probability'] = 1

    for row in range(final_scenarios):
        # print(row)

        branch = output.iloc[[row]][0][0]
        # print(branch)

        for i in range(1, forecast_steps+1):
            # print(i)
            output.loc[branch, i] = float(scenarios.loc[scenarios['node'] == branch[0:i]][instrument])
            output.loc[branch, 'probability'] = output.loc[branch, 'probability'] * float(scenarios.loc[scenarios['node'] == branch[0:i]]['probability'])

    # Put to plottable format
    output_plot = output.T

    # Change the first row to 0th time period price
    output_plot.iloc[[0]] = data.tail(1)[instrument][0]

    # Drop the last row and give cumulative sum
    output_plot = output_plot[:-1]
    output_cum = output_plot.astype(float).cumprod()


    # Plot
    if to_plot == 'yes':
        plt.style.use("seaborn-darkgrid")
        plot_title = ('Time-series of %s Stock Price Simulation' %instrument)
        ax = output_cum.plot(legend=False, title=plot_title, colormap='ocean')
        ax.set_xlabel("Forecast Step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Price ($)")

    return output, output_cum


def cluster_output_modifier(instruments, scenarios, branching):

    # Define dataframe size
    forecast_steps = len(branching)
    nr_scenarios = reduce(lambda x, y: x*y, branching[1:])
    nr_trees = branching[0]

    # Allocate the scenarios to trees based on the first branching layer and separate the assets
    scenario_dict = {}
    for k in range(nr_trees):
        # print(k)
        k_tree_data = scenarios.loc[scenarios.node.str[0].eq(str(i))] # Gets the data specific to the tree

        instrument_dict = {}
        for instrument in instruments:
            # print(instrument)

            output = pd.DataFrame(np.random.randint(low=1, high=10, size=(nr_scenarios, forecast_steps + 1)))
            output[0] = list(k_tree_data[-nr_scenarios:]['node'])
            output.index = k_tree_data[-nr_scenarios:]['node']

            output['probability'] = 1

            for s in range(nr_scenarios):
                # print(row)

                branch = output.iloc[[s]][0][0]
                # print(branch)

                for t in range(1, forecast_steps+1):
                    # print(i)
                    output.loc[branch, t] = float(k_tree_data.loc[scenarios['node'] == branch[0:t]][instrument])
                    output.loc[branch, 'probability'] = output.loc[branch, 'probability'] * float(k_tree_data.loc[k_tree_data['node'] == branch[0:t]]['probability'])

            # Amend the column names
            output.columns = ['node', *range(1, forecast_steps+1), 'probability']

            # Add to the instruments dictionary
            instrument_dict[instrument] = output

        scenario_dict[str(k)] = instrument_dict

    return scenario_dict

####################### END ########################
