#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################
# IMPORTS GO HERE
####################################################
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

####################################################
# SOME PARAMETERS
####################################################
my_cmap = cm.hot #cm.gray # Or choose cm.rainbow, cm.hot, cm.winter, etc.

def plot(u):
  plt.figure(figsize=(8,8))
  img = plt.imshow(u, cmap=my_cmap, interpolation='nearest', origin='lower')
  plt.colorbar(img)
  plt.show()

def animate(all_sims):
  NumSims = all_sims.shape[0] 
  fig, ax = plt.subplots()
  img = plt.imshow(all_sims[0,:,:], cmap=my_cmap, interpolation='nearest', origin='lower')
  plt.colorbar(img)
  def animate(i):
    img.set_data(all_sims[i,:,:])
    return img,

  #Init only required for blitting to give a clean slate.
  def init():
      img.set_data(all_sims[0,:,:])
      return img,

  ani = animation.FuncAnimation(fig, animate, np.arange(1, NumSims), init_func=init,
      interval=100, blit=True)
  plt.show()
