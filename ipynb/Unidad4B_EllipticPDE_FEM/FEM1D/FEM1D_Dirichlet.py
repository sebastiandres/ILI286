import numpy as np
from matplotlib import pyplot
from numpy.linalg import solve, norm, lstsq

import Common

from pdb import set_trace as st

def DirichletSolver(f, u0, u1, N):
  '''
  Computes the solution using Dirichlet Boundary Conditions:
  u(0) = u(1) = 0
  '''
  # PARAMETERS
  x = np.linspace(0., 1., N)
  h = x[1]-x[0]
  # TRIDIAG MATRIX
  K = (1./h)*(  np.diag(-1*np.ones(N-1),-1) 
               +np.diag( 2*np.ones(N), 0) 
               +np.diag(-1*np.ones(N-1),+1)
               )
  b = h*f(x)
  # SOLUTION
  u = solve(K, b)
  return x, u


f1 = lambda x : np.e**x
u1 = lambda x : -np.e**x + (np.e - 1.)*x + 1.

f2 = lambda x : np.sin(x) - x 
u2 = lambda x : np.sin(x) - x*np.sin(1.) +  (x**3 - x )/6. 

bc_u0 = 0.
bc_u1 = 0.

# Example
f, u = f2, u2
N = 8
x, u_h = DirichletSolver(f, bc_u0, bc_u1, N)
Common.SolutionPlot(x, u, u_h)

"""
# Convergence
f, u = f2, u2

N_list = 64.*2**np.arange(6)
h_list = []
error_list = []
for N in N_list:
  x, u_h = DirichletSolver(f, bc_u0, bc_u1, N)
  h = x[1]-x[0]
  h_list.append(h)
  error_list.append( Common.Error(x, u, u_h) )

Common.ErrorPlot(h_list, error_list)
"""
