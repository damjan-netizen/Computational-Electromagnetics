# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:54:43 2022

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

edge = np.linspace(-1,1,300)

upper_y = np.cos(np.pi*edge/2)
lower_y = edge**4
upper_x = 1/(np.e**-1-np.e)*(np.exp(edge) - np.e)
lower_x = 0.5*(edge**2-edge)


#define the meshgrid

xx, yy = np.meshgrid(edge, edge)


#define function to solve the potential

@numba.jit("f8[:,:](f8[:,:], i8)", nopython=True, nogil=True)
def compute_potential(potential, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1,length-1):
            for j in range(1,length-1):
                potential[i][j] = 1/4*(potential[i+1,j] + potential[i-1,j] +
                                       potential[i,j+1]+potential[i,j-1])
             
    return potential




"""
Solve for potential
"""

potential = np.zeros((300,300))

#set the boundary conditions

potential[0,:] = lower_y
potential[-1,:] = upper_y
potential[:,0] = lower_x
potential[:,-1] = upper_x


potential = compute_potential(potential, n_iter= 10000)

# plt.contourf(xx,yy,potential, levels = 35)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title(r'Contour plot of the solution to Laplace eqn')
# plt.tick_params(axis='both', labelsize=7)
# plt.colorbar(label='Voltage [V]')

"""
WITH A BLOCK OF FIXED POTENTIAL
"""

def potential_block(x, y):
    return np.select([(x>0.5)*(x<0.7)*(y>0.5)*(y<0.7),
                      (x<=0.5)+(x>=0.7)+(y<=0.5)+(y>=0.7)],[1,0])



fixed_pot = potential_block(xx,yy)

#get the locations where we put the potential block
fixed_bool = fixed_pot != 0

#define function to solve the potential

@numba.jit("f8[:,:](f8[:,:],b1[:,:], i8)", nopython=True, nogil=True)
def compute_potential(potential,fixed_bool, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1,length-1):
            for j in range(1,length-1):
                if(not fixed_bool[i,j]):
                    potential[i][j] = 1/4*(potential[i+1,j] + potential[i-1,j] +
                                           potential[i,j+1]+potential[i,j-1])
             
    return potential

#set the boundary conditions

potential = np.zeros((300,300))
potential[0,:] = lower_y
potential[-1,:] = upper_y
potential[:,0] = lower_x
potential[:,-1] = upper_x
potential[fixed_bool] = fixed_pot[fixed_bool]

potential = compute_potential(potential, fixed_bool, 10000)

# plt.figure(figsize=(8,6))
# plt.contourf(xx,yy,potential, levels = 35)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title(r'Solution of The Laplace eqn with a fixed potential block')
# plt.tick_params(axis='both', labelsize=7)
# plt.colorbar(label='Voltage [V]')

#to plot the electric field intensity

Ex, Ey = np.gradient(-potential)

E_mag = np.sqrt(Ex**2 + Ey**2)


# plt.figure(figsize=(8,6))
# plt.contourf(xx,yy,E_mag, levels = 35)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# plt.title(r'Electric field intensity')
# plt.tick_params(axis='both', labelsize=7)
# plt.colorbar(label='Electric field [V/cm]')

fig, axes = plt.subplots(figsize=(14,6), nrows=1, ncols=2)

ax = axes[0]
im1 = ax.contourf(xx,yy,potential, levels = 35)
ax.set_xlabel(r"$x$",fontsize=7)
ax.set_ylabel(r"$y$",fontsize=7)
ax.set_title(r'Solution of The Laplace eqn with a fixed potential block')
ax.tick_params(axis='both', labelsize=7)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax = axes[1]

im2 = ax.contourf(xx,yy,E_mag, levels = 35)
ax.set_xlabel(r"$x$",fontsize=7)
ax.set_ylabel(r"$y$",fontsize=7)
ax.set_title(r'Electric field intensity')
ax.tick_params(axis='both', labelsize=7)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')


img = color.rgb2gray(io.imread('accordion_cap.png'))


