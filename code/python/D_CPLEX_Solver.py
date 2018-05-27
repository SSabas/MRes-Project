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

import cplex


# ------------------------------ DEFINE THE PROGRAM ------------------------------- #



# Test


import cplex

# Create an instance of a linear problem to solve
problem = cplex.Cplex()


# We want to find a maximum of our objective function
problem.objective.set_sense(problem.objective.sense.maximize)

# The names of our variables
names = ["x", "y", "z"]

# The obective function. More precisely, the coefficients of the objective
# function. Note that we are casting to floats.
objective = [5.0, 2.0, -1.0]

# Lower bounds. Since these are all zero, we could simply not pass them in as
# all zeroes is the default.
lower_bounds = [0.0, 0.0, 0.0]

# Upper bounds. The default here would be cplex.infinity, or 1e+20.
upper_bounds = [100, 1000, cplex.infinity]

problem.variables.add(obj = objective,
                      lb = lower_bounds,
                      ub = upper_bounds,
                      names = names)

# Constraints

# Constraints are entered in two parts, as a left hand part and a right hand
# part. Most times, these will be represented as matrices in your problem. In
# our case, we have "3x + y - z ≤ 75" and "3x + 4y + 4z ≤ 160" which we can
# write as matrices as follows:

# [  3   1  -1 ]   [ x ]   [  75 ]
# [  3   4   4 ]   [ y ] ≤ [ 160 ]
#                  [ z ]

# First, we name the constraints
constraint_names = ["c1", "c2"]

# The actual constraints are now added. Each constraint is actually a list
# consisting of two objects, each of which are themselves lists. The first list
# represents each of the variables in the constraint, and the second list is the
# coefficient of the respective variable. Data is entered in this way as the
# constraints matrix is often sparse.

# The first constraint is entered by referring to each variable by its name
# (which we defined earlier). This then represents "3x + y - z"
first_constraint = [["x", "y", "z"], [3.0, 1.0, -1.0]]
# In this second constraint, we refer to the variables by their indices. Since
# "x" was the first variable we added, "y" the second and "z" the third, this
# then represents 3x + 4y + 4z
second_constraint = [[0, 1, 2], [3.0, 4.0, 4.0]]
constraints = [ first_constraint, second_constraint ]

# So far we haven't added a right hand side, so we do that now. Note that the
# first entry in this list corresponds to the first constraint, and so-on.
rhs = [75.0, 160.0]

# We need to enter the senses of the constraints. That is, we need to tell Cplex
# whether each constrains should be treated as an upper-limit (≤, denoted "L"
# for less-than), a lower limit (≥, denoted "G" for greater than) or an equality
# (=, denoted "E" for equality)
constraint_senses = [ "L", "L" ]

# Note that we can actually set senses as a string. That is, we could also use
#     constraint_senses = "LL"
# to pass in our constraints

# And add the constraints
problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

# Solve the problem
problem.solve()

# And print the solutions
print(problem.solution.get_values())


###################################

# Create an instance of a linear problem to solve
problem = cplex.Cplex()


# We want to find a maximum of our objective function
problem.objective.set_sense(problem.objective.sense.maximize)

# The names of our variables
names = ["x", "y", "z"]
objective = [5.0, 2.0, -1.0]
lower_bounds = [0.0, 0.0, 0.0]
upper_bounds = [100, 1000, cplex.infinity]

problem.variables.add(obj = objective,
                      lb = lower_bounds,
                      ub = upper_bounds,
                      names = names)

constraint_names = ["c1", "c2"]

first_constraint = [["x", "y", "z"], [3.0, 1.0, -1.0]]

second_constraint = [[0, 1, 2], [3.0, 4.0, 4.0]]
constraints = [ first_constraint, second_constraint ]
rhs = [75.0, 160.0]

# We need to enter the senses of the constraints. That is, we need to tell Cplex
# whether each constrains should be treated as an upper-limit (≤, denoted "L"
# for less-than), a lower limit (≥, denoted "G" for greater than) or an equality
# (=, denoted "E" for equality)
constraint_senses = [ "L", "L" ]

# And add the constraints
problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

# Solve the problem
problem.solve()

# And print the solutions
print(problem.solution.get_values())

##################################

# ============================================================
# This file gives us a sample to use Cplex Python API to
# establish a Linear Programming model and then solve it.
# The Linear Programming problem displayed bellow is as:
#                  min z = cx
#    subject to:      Ax = b
# ============================================================

# ============================================================
# Input all the data and parameters here
num_decision_var = 3
num_constraints = 3

A = [
    [1.0, -2.0, 1.0],
    [-4.0, 1.0, 2.0],
    [-2.0, 0, 1.0],
]
b = [11.0, 3.0, 1.0]
c = [-3.0, 1.0, 1.0]

constraint_type = ["L", "G", "E"] # Less, Greater, Equal
# ============================================================

# Establish the Linear Programming Model
myProblem = cplex.Cplex()

# Add the decision variables and set their lower bound and upper bound (if necessary)
myProblem.variables.add(names= ["x"+str(i) for i in range(num_decision_var)])
for i in range(num_decision_var):
    myProblem.variables.set_lower_bounds(i, 0.0)

# Add constraints
for i in range(num_constraints):
    myProblem.linear_constraints.add(
        lin_expr= [cplex.SparsePair(ind= [j for j in range(num_decision_var)], val= A[i])],
        rhs= [b[i]],
        names = ["c"+str(i)],
        senses = [constraint_type[i]]
    )

# Add objective function and set its sense
for i in range(num_decision_var):
    myProblem.objective.set_linear([(i, c[i])])
myProblem.objective.set_sense(myProblem.objective.sense.minimize)

# Solve the model and print the answer
myProblem.solve()
print(myProblem.solution.get_values())