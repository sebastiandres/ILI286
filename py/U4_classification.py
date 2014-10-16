from matplotlib import pyplot as plt
import numpy as np

# Define the parameters
A = 1.0
B = 1.0 # Test 1.0, 2.0, 3.0
C = 1.0
D = 2.0
E = 0.0
G =-10.0 # F not needed
xmin, xmax = -10.0, 10.0
ymin, ymax = -10.0, 10.0
Nx, Ny = 1000, 1000
Delta = B**2-4*A*C
if Delta<0:  print "Elliptic Equation"
if Delta==0: print "Parabolic Equation"
if Delta>0:  print "Hiperbolic Equation"

xrange = np.linspace(xmin, xmax, Nx)
yrange = np.linspace(ymin, ymax, Ny)
X, Y = np.meshgrid(xrange,yrange)

# F is one side of the equation, G is the other
EQN = A*X*X + B*X*Y + C*Y*Y + D*X + E*Y + G 

plt.contour(X, Y, EQN, [0.0], linewidths=[2.0])
plt.grid('on')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
