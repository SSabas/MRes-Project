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
from scipy.optimize import curve_fit

# -------------------- CURVE FITTING ----------------------------- #


# Define a function for the exponential to be fitted
def exp_function(x, a, b):
    return a * np.exp(-b * x)


# Fit the curve and get the residuals
def fit_exp_function(exp_function, data, to_plot='yes'):

    # Extract relevant series
    ydata = np.array(data)
    xdata = range(1, len(ydata)+1)

    # Fit the curve (extract the parameters a and b)
    parameters, parameters_cov = curve_fit(exp_function, xdata, ydata)

    # Plot the results
    if to_plot == 'yes':
        plt.plot(data.index, ydata, 'b-', label='Data')
        plt.plot(data.index, exp_function(xdata, *parameters), 'r-', label='Fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    # Get the residuals
    residuals = ydata - exp_function(xdata, *parameters)

    return residuals, parameters


# Calculate the moments for scenario simulation
def calculate_moments(exp_function, data, to_plot='no'):

    # Dictionary placeholders
    means = {}
    variances = {}

    # Calculate the exponential growths
    for column in data:
        print(column)

        price_series = data[column]
        residuals, parameters = fit_exp_function(exp_function, price_series, to_plot=to_plot)
        means[column] = -parameters[1]

    # Calculate the (co-)variances
    for column_j in data:
        price_series_j = data[column_j]
        M_j = np.exp(means[column_j])*np.array(price_series.tail(1))

        # Placeholder for variances
        variances[column_j] = []

        for column_l in data:
            price_series_l = data[column_l]
            M_l = np.exp(means[column_l]) * np.array(price_series.tail(1))
            C_jl = (1/(data.shape[0]-2+1)) * np.sum((price_series_j-M_j)*(price_series_l-M_l))
            variances[column_j].append(C_jl)


    # Put the covariances to K*K matrix
    variance_matrix = []
    mean_matrix = []

    for column_j in data:
        mean_matrix.append(means[column_j])
        variance_matrix.append(variances[column_j])

    variance_matrix = np.array(variance_matrix).reshape(data.shape[1], data.shape[1])

    return mean_matrix, variance_matrix

#calculate_moments(exp_function, data)


# Output in format required for C++ simulation
def cpp_layout(path, data, exp_function, branching=(4, 4, 4)):

    # Get means and variances
    mean_matrix, variance_matrix = calculate_moments(exp_function, data)

    with open(path, "w") as text_file:
        print("ASSETS %d" %(data.shape[1]), file=text_file)
        print("STAGES %d" %(len(branching)), file=text_file)
        print("COVARIANCES", file=text_file)
        n = 1
        for i in variance_matrix:
            print(" ".join(map(str, i[0:n])), file=text_file)
            n += 1
        print("INITIAL", file=text_file)
        print(" ".join(map(str, np.array(data.tail(1))[0])), file=text_file)
        print("GROWTH", file=text_file)
        print(" ".join(map(str, mean_matrix)), file=text_file)
        print("BRANCHING", file=text_file)
        print(" ".join(map(str, branching)), file=text_file)
        print("END", file=text_file)

# path = os.getcwd() + "/code/cpp/cluster2/sven_test.txt"
# cpp_layout(path, data, exp_function, branching=(4, 4, 4))

####################### END ########################

