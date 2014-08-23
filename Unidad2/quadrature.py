import numpy as np
from matplotlib import pyplot as plt

from pdb import set_trace as st

# Numerical Quadrature examples

def plot(f, xbin, ybin):
  plt.figure(figsize=(8,10))
  N = 201
  # Get a representation of f as a continuous function
  x = np.linspace(xbin.min(), xbin.max(), N)
  y = f(x)
  # Plot it all
  plt.plot(x, y, 'r', lw=2.0)
  plt.fill_between(xbin, 0, ybin, alpha=0.5)
  # Setting the lims
  ymin, ymax = y.min(), y.max()
  dy = .1*(ymax-ymin)
  plt.ylim([ymin-dy,ymax+dy])
  xmin, xmax = x.min(), x.max()
  dx = .1*(xmax-xmin)
  plt.xlim([xmin-dx,xmax+dx])
  plt.grid("on")
  plt.show()
  return

def midpoint(myfun, N, xmin, xmax, do_plot=True):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points  
  dx = x[1]-x[0]
  midpoints = x[:-1] + .5*dx
  midpoint_values = f(midpoints)
  int_val = sum(midpoint_values*dx)
  if do_plot:
    xbin = np.vstack([x[:-1], x[1:]]).flatten(1)
    ybin = np.vstack([midpoint_values, midpoint_values]).flatten(1)
    plot(f, xbin, ybin)
  return int_val

def trapezoid(myfun, N, xmin, xmax, do_plot=True):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points  
  dx = x[1]-x[0]
  xleft = x[:-1]
  xright = x[1:]
  int_val = sum(0.5*(f(xleft)+f(xright))*dx)
  if do_plot:
    xbin = x
    ybin = f(x) 
    plot(f, xbin, ybin)
  return int_val

if __name__=="__main__":
  N = 40
  xmin = -1
  xmax = 1
  myfun = lambda x :  np.sin(x)/x #1.0 if abs(x)<1E-6  else np.sin(x)/x
  print midpoint(myfun, N, xmin, xmax)
  print trapezoid(myfun, N, xmin, xmax)
