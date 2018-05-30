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

####################### END ########################
