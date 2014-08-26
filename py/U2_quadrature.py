import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rc('savefig', bbox="tight")

from pdb import set_trace as st
from IPython import embed as ip

###########################################################################
# General plotting framework
###########################################################################
def plot(f, xbin, ybin, int_val, N, text, figname=""):
  plt.figure(figsize=(10,8))
  n = 201
  # Get a representation of f as a continuous function
  x = np.linspace(xbin.min(), xbin.max(), n)
  y = f(x)
  # Plot the function
  plt.plot(x, y, 'r', lw=2.0)
  # Plot the interpolation
  plt.fill_between(xbin, 0, ybin, alpha=0.25, lw=2.0)
  # Setting the lims
  ymin, ymax = y.min(), y.max()
  dy = .1*(ymax-ymin)
  plt.ylim([ymin-dy,ymax+dy])
  xmin, xmax = x.min(), x.max()
  dx = .1*(xmax-xmin)
  plt.xlim([xmin-dx,xmax+dx])
  # Do the text
  if N>1:
    text_N = r"$%s \approx %.5f$ (usando %d evaluaciones de $f$)" %(text, int_val, N)
    plt.text(min(x), max(y), text_N, fontsize=18)
    plt.text(min(x), 0.9*max(y), "Valor exacto $2.35040$", fontsize=18)
  plt.xlabel("x")
  plt.ylabel("y")
  if not figname:
    plt.show()
  else:
    plt.savefig(figname)
    plt.close()
  return

###########################################################################
# Riemann Rule
###########################################################################
def riemann(myfun, N, xmin, xmax, direction="left", do_plot=True, text="", figname=""):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points  
  dx = x[1]-x[0]
  if direction=="left":
    points = x[:-1]
  elif direction=="right":
    points = x[1:]
  else:
    print "Riemann Sum: choose left or right"
    return
  point_values = f(points)
  int_val = sum(point_values*dx)
  if do_plot:
    xbin = np.vstack([x[:-1], x[1:]]).flatten(1)
    ybin = np.vstack([point_values, point_values]).flatten(1)
    plot(f, xbin, ybin, int_val, N, text, figname)
  return int_val

###########################################################################
# Midpoint Rule
###########################################################################
def midpoint(myfun, N, xmin, xmax, do_plot=True, text="", figname=""):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points  
  dx = x[1]-x[0]
  midpoints = x[:-1] + .5*dx
  midpoint_values = f(midpoints)
  int_val = sum(midpoint_values*dx)
  if do_plot:
    xbin = np.vstack([x[:-1], x[1:]]).flatten(1)
    ybin = np.vstack([midpoint_values, midpoint_values]).flatten(1)
    plot(f, xbin, ybin, int_val, N, text, figname)
  return int_val

###########################################################################
# Trapezoid Rule
###########################################################################
def trapezoid(myfun, N, xmin, xmax, do_plot=True, text="", figname=""):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points  
  dx = x[1]-x[0]
  xleft = x[:-1]
  xright = x[1:]
  int_val = sum(0.5*(f(xleft)+f(xright))*dx)
  if do_plot:
    xbin = x
    ybin = f(x) 
    plot(f, xbin, ybin, int_val, N, text, figname)
  return int_val

###########################################################################
# Simpsons Rule
###########################################################################
def simpsons(myfun, N, xmin, xmax, do_plot=True, text="", figname=""):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  x = np.linspace(xmin, xmax, N+1) # We want N bins, so N+1 points
  if N%2==1:
    print "Simpsons rule only applicable to even number of segments"
    return
  dx = x[1]-x[0]
  xleft   = x[:-2:2]
  xmiddle = x[1::2]
  xright  = x[2::2]
  int_val = sum((f(xleft)+4*f(xmiddle)+f(xright))*dx/3)
  if do_plot:
    xbin, ybin = simpsons_bins(f, xleft, xmiddle, xright)
    plot(f, xbin, ybin, int_val, N, text, figname)
  return int_val

def simpsons_bins(f, xleft, xmiddle, xright):
  xbin, ybin = [], []
  n = 21
  for x0, x1, x2 in zip(xleft, xmiddle, xright):
    x = np.linspace(x0, x2, n)
    y = (f(x0)*(x-x1)*(x-x2)) / ((x0-x1)*(x0-x2))
    y+= (f(x1)*(x-x0)*(x-x2)) / ((x1-x0)*(x1-x2))
    y+= (f(x2)*(x-x0)*(x-x1)) / ((x2-x0)*(x2-x1))
    xbin.extend(list(x))
    ybin.extend(list(y))
  return np.array(xbin), np.array(ybin)

###########################################################################
# Simpsons Rule
###########################################################################
def gaussianquad(myfun, N, xmin, xmax, do_plot=True, text="", figname=""):
  f = np.vectorize(myfun) # So we can apply it to arrays without trouble
  if N==1:
    x = np.array([1])
    w = np.array([2])
  elif N==2:
    x = np.array([-0.577, 0.577])
    w = np.array([1, 1])
  elif N>=4:
    x = np.array([-0.861, -0.339, 0.339, 0.861])
    w = np.array([0.348, 0.652, 0.652, 0.348])
  x, w = gaussian_nodes_and_weights(N)
  int_val = sum( w * f(x) )
  if do_plot:
    xbin, ybin = gaussian_bins(f, x, w)
    plot(f, xbin, ybin, int_val, N, text, figname)
  return int_val

def gaussian_nodes_and_weights(N):
  if N==1: return np.array([1]), np.array([2])
  beta = .5 / np.sqrt(1.-(2.*np.arange(1.,N))**(-2))
  T = np.diag(beta,1) + np.diag(beta,-1)
  D, V = np.linalg.eigh(T)
  x = D
  w = 2*V[0,:]**2
  return x, w

def gaussian_bins(f, x, w):
  z = [xmin] + list(xmin+w.cumsum())
  xbin = np.vstack([z[:-1], z[1:]]).flatten(1)
  z = f(x)
  ybin = np.vstack([z[:], z[:]]).flatten(1)
  return np.array(xbin), np.array(ybin)

###########################################################################
if __name__=="__main__":
  xmin = -1
  xmax = 1
  myfun = lambda x : np.exp(x)
  N_values = [1, 2, 4, 8, 16, 32, 64, 128, 1024]
  text= r"\int_{-1}^{+1} e^x dx"
  for N in N_values:
    print riemann(myfun, N, xmin, xmax, direction="left", 
                  text=text, figname="riemann_left_%d.png"%N)
    print riemann(myfun, N, xmin, xmax, direction="right", 
                  text=text, figname="riemann_right_%d.png"%N)
    print midpoint(myfun, N, xmin, xmax, 
                   text=text, figname="midpoint_%d.png"%N)
    print trapezoid(myfun, N, xmin, xmax, 
                    text=text, figname="trapezoid_%d.png"%N)
    print simpsons(myfun, N, xmin, xmax, 
                   text=text, figname="simpsons_%d.png"%N)
    print gaussianquad(myfun, N, xmin, xmax, 
                       text=text, figname="gaussianquad_%d.png"%N)

