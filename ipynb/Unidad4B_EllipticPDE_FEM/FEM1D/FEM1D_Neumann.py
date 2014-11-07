from numpy import *
from matplotlib import pylab
from numpy.linalg import solve, norm, lstsq
from math import e, pi,
from pdb import set_trace as st

def NewmannBoundaryConditions(N, f, sol, plot=False):
  '''
  Computes the solution using Dirichlet Boundary Conditions:
  u(0) = u'(1) = 1
  '''
  # PARAMETERS
  h = 1./N
  # TRIDIAG MATRIX
  K = (1./h)*( diag(-1*ones(N-1),-1) 
                 + diag( 2*ones(N), 0) 
                 + diag(-1*ones(N-1),+1)
               )
  K[-1,-2:] = [-1/h, 1/h];
  x = arange(1.,N+1.)/N
  L = h*f(x)
  L[-1] = h/2.*f(1.0) + 1.0
  # SOLUTION
  U = solve(K,L)
  # PLOT
  x_ext = arange(0.,N+1.)/N
  z = array([0.])
  U_ext = concatenate([z,U]) + 1 # Correction
  if plot==True:
    h_sol = min(h/2,0.001)
    x_sol = arange(0.,1.+h_sol,h_sol)
    U_sol = sol(x_sol)
    pylab.figure(figsize=(10,6))
    pylab.plot(x_ext, U_ext, 'rx', label='u_h(x)', lw=1.5, mew=1.5)
    pylab.plot(x_sol, U_sol, 'k',  label='u(x)', lw=1.5)
    pylab.legend(loc=0)
    pylab.xlabel('x')
    pylab.ylabel('f(x)')
    pylab.savefig('DN%sN%d.png' %(f.func_name,N))
    pylab.figure(figsize=(10,6))
    pylab.plot(x_ext, sol(x_ext)- U_ext, 'rx:', label='u - u_h', lw=1.5, mew=1.5)
    pylab.xlabel('x')
    pylab.ylabel('Pointwise error')
    pylab.legend(loc=0)
    pylab.gca().yaxis.set_major_formatter(pylab.FormatStrFormatter('%1.2E'))
    pylab.savefig('DNdiff%sN%d.png' %(f.func_name,N))
  return norm(sol(x_ext)-U_ext,2)*h**.5

def Error(f, sol, Nmin, Nmax, label):
  dN = 10
  N_list = arange(Nmin,Nmax+10,10)
  error = zeros(len(N_list));
  for n,N in enumerate(N_list):
    if label=='DD':
      error[n] = DirichletBoundaryConditions(N, f, sol)
    elif label=='DN':
      error[n] = NewmannBoundaryConditions(N, f, sol)
  # Error N plot
  pylab.figure(figsize=(10,6))
  pylab.plot(N_list, error, 'rx--', label='$||u-u_h||_{L^2}$', lw=1.5, mew=1.5)
  pylab.xlabel('N')
  pylab.ylabel('$L^2$ norm error')
  pylab.legend(loc=0)
  pylab.gca().yaxis.set_major_formatter(pylab.FormatStrFormatter('%1.2E'))
  pylab.savefig(label+'ErrorN'+f.func_name+'.png')
  # Error h plot
  h = 1./N_list
  pylab.figure(figsize=(10,6))
  pylab.plot(h, error, 'gx--', label='$||u-u_h||_{L^2}$', lw=1.5, mew=1.5)
  pylab.xlabel('h')
  pylab.ylabel('$L^2$ norm error')
  pylab.legend(loc=0)
  pylab.gca().yaxis.set_major_formatter(pylab.FormatStrFormatter('%1.2E'))
  pylab.savefig(label+'Errorh'+f.func_name+'.png')
  # log Error h plot
  log_h = log(h)
  A = concatenate([matrix(log(h)).T,ones([len(h),1])],axis=1)
  log_e = log(error)
  m, c = lstsq(A, log_e)[0]
  pylab.figure(figsize=(10,6))
  pylab.plot(log_h, log_e, 'gx',label='$||u-u_h||_{L^2}$', lw=1.5, mew=1.5)
  pylab.plot(log_h, m*log_h+c, 'k-',label='%1.2f log(h) + %1.2f' %(m,c))
  pylab.xlabel('log h')
  pylab.ylabel('$L^2$ norm log error')
  pylab.legend(loc=0)
  pylab.savefig(label+'loglogError'+f.func_name+'.png')
  # Least squares fit of the loglog
  return h, error

################################################################################
# GENERAL FUNCTIONS
def f1(x) : return e**x
def f2(x) : return sin(x) - x 

################################################################################
# NEWMANN

def sol1DN(x) : return -e**x + (e+1)*x + 2
def sol2DN(x) : return sin(x) + x**3/6. + x*(1./2. - cos(1.) ) + 1.

print 'Newmann Boundary Conditions'
N = 100
print 'Error N=%d :%e  ' %( N, NewmannBoundaryConditions(N, f1, sol1DN, plot=True) )
h, error = Error(f1,sol1DN,10,1000,'DN')

print 'Error N=%d :%e  ' %( N, NewmannBoundaryConditions(N, f2, sol2DN, plot=True) )
h, error = Error(f2,sol2DN,10,1000,'DN')
