#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################
# IMPORTS GO HERE
####################################################
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def show(x, t, u):
  Nt = len(t) 
  fig, ax = plt.subplots()
  l, = plt.plot(x, u[:,0],'ko:')
  dx = 0.05*(x[-1]-x[0])
  ymin = u.min().min()
  ymax = u.max().max()
  dy = 0.05*(ymax-ymin)
  ax.set_xlim([x[0]-dx, x[-1]+dx])
  ax.set_ylim([ymin-dy, ymax+dy])
  plt.xlabel("$x$", fontsize=20)
  plt.ylabel("$u(x,t)$", fontsize=20)
  def animate(i):
      l.set_ydata(u[:,i])
      return l,

  #Init only required for blitting to give a clean slate.
  def init():
      l.set_ydata(np.ma.array(u[:,0], mask=True))
      return l,

  dt = t[1]-t[0]
  #interval = 4 * 100. * 200/Nt   # So simulations run in the same time regardless of Nt
  interval = 100
  ani = animation.FuncAnimation(fig, animate, np.arange(1, Nt), init_func=init,
      interval=interval, blit=True)

  plt.show()
