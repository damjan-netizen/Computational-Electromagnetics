# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:12:19 2022

@author: GroH Von Hilfiger
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib import cm
from skimage import color
from skimage import io
import numba
from numba import jit
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Boundary conditions
"""
img = color.rgb2gray(io.imread('capacitor.png'))
dimX, dimY = img.shape

edgeX = np.linspace(-1,1,dimY)
edgeY = np.linspace(-1,1,dimX)

# upper_y = np.cos(np.pi*edgeX/2)
# lower_y = edgeX**4
# upper_x = 1/(np.e**-1-np.e)*(np.exp(edgeY) - np.e)
# lower_x = 0.5*(edgeY**2-edgeY)

upper_y = 0*edgeX
lower_y = 0*edgeX
upper_x = 0*edgeY
lower_x = 0*edgeY


#define the meshgrid

xx, yy = np.meshgrid(edgeX, edgeY)


#define the computational function

@numba.jit("f8[:,:](f8[:,:],b1[:,:], i8)", nopython=True, nogil=True)
def compute_potential(potential,fixed_bool, n_iter):
    lengthX, lengthY = potential.shape
    for n in range(n_iter):
        for i in range(1,lengthX-1):
            for j in range(1,lengthY-1):
                if(not fixed_bool[i,j]):
                    potential[i][j] = 1/4*(potential[i+1,j] + potential[i-1,j] +
                                           potential[i,j+1]+potential[i,j-1])
             
    return potential



#Set up the copmputation conditions

HV_bool = img<0.40
LV_bool = img>0.86
fixed_bool = HV_bool + LV_bool
fixed_pot = img

plt.contourf(img)
plt.colorbar()

potential = np.zeros((dimX,dimY))
potential[0,:] = lower_y
potential[-1,:] = upper_y
potential[:,0] = lower_x
potential[:,-1] = upper_x
potential[fixed_bool] = fixed_pot[fixed_bool]

##Solve for potential
potential = compute_potential(potential, fixed_bool, 10000)

## Solve for the electric field

Ex, Ey = np.gradient(-potential)

E_mag = np.sqrt(Ex**2 + Ey**2)


fig, axes = plt.subplots(figsize=(14,6), nrows=1, ncols=2)

ax = axes[0]
im1 = ax.contourf(xx, yy, potential, levels=50)
ax.set_xlabel(r'$x$', fontsize=7)
ax.set_ylabel(r'$y$', fontsize=7)
ax.set_title('Map of electric potential')
ax.tick_params(axis='both', labelsize=7)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax = axes[1]
im2 = ax.contourf(xx, yy, E_mag, levels=50)
ax.set_xlabel(r'$x$', fontsize=7)
ax.set_ylabel(r'$y$', fontsize=7)
ax.set_title('Map of electric field strength')
ax.tick_params(axis='both', labelsize=7)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right',size='5%',pad=0.05)
fig.colorbar(im2, cax=cax, orientation = 'vertical')

