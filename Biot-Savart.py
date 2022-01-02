# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 21:07:39 2021

@author: GroH Von Hilfiger
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import plotly.graph_objects as go
from IPython.display import HTML
import sympy as smp
from sympy.vector import cross

from plotly.offline import plot



phi = np.linspace(0, 2*np.pi, 100)


#dimensionless l, with R==1
#current is going in ccw direction

def l(phi):
    return (1 + 3/4 * np.sin(3*phi))*np.array([np.cos(phi), np.sin(phi), np.zeros(len(phi))])

lx, ly, lz = l(phi)

plt.figure()
plt.plot(lx,ly)
plt.xlabel(r'$x/R$')
plt.ylabel(r'$y/R$')

#### Now to get the expressions for dl/dphi, vec{r-l}, all using sympy

t, x, y, z = smp.symbols('varphi, x, y, z')

#get l, r and the separation vector r-l

l = (1 + 3/4 * smp.sin(3*t))*smp.Matrix([smp.cos(t), smp.sin(t), 0])

r = smp.Matrix([x, y, z])

sep = r - l


### Define the integrand in symbolic form, t stands for phi

integrand = smp.diff(l,t).cross(sep) / sep.norm()**3

#now we have to go from the symbolic space to the numeric space

#get the x, y, z components of the integrand

dBxdt = smp.lambdify([t, x, y, z], integrand[0])
dBydt = smp.lambdify([t, x, y, z], integrand[1])
dBzdt = smp.lambdify([t, x, y, z], integrand[2])


## Get the integral by performing the integral over each component

def B(x, y, z):
    return np.array([quad(dBxdt, 0, 2*np.pi, args=(x,y,z))[0],
                     quad(dBydt, 0, 2*np.pi, args=(x,y,z))[0],
                     quad(dBzdt, 0 ,2*np.pi, args=(x,y,z))[0]])


####### TO LOOK AT THE VECTOR FIELD #####


# set up a meshgrid to solve for the field in some 3D volume

x = np.linspace(-2, 2, 20)

xx, yy, zz = np.meshgrid(x,x,x)

#vectorize passes the arguments to B(x,y,z) point by point, since it is all quad() can process
#we need to do this since xx, yy, zz are 3D arrays
 
B_field = np.vectorize(B, signature='(),(),()->(n)')(xx,yy,zz)

Bx = B_field[:,:,:,0]
By = B_field[:,:,:,1]
Bz = B_field[:,:,:,2]

plt.figure()
plt.hist(Bx.ravel(), bins=100,histtype='step',label='Bx')
plt.hist(By.ravel(), bins=100,histtype='step',label='By')
plt.hist(Bz.ravel(), bins=100,histtype='step',label='Bz')
plt.legend(loc=0)
plt.xlabel('Magnetic field magnitude')
plt.ylabel('Frequency')

#to keep B field from blowing up when close to the wire
Bx[Bx>20] = 20
By[By>20] = 20
Bz[Bz>20] = 20

Bx[Bx<-20] = -20
By[By<-20] = -20
Bz[Bz<-20] = -20

##use plotly to make an interactive 3D plot

data = go.Cone(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
                u=Bx.ravel(), v=By.ravel(), w=Bz.ravel(),
                colorscale='Inferno', colorbar=dict(title=r'$x^2$'),
                sizemode="absolute", sizeref=20)

layout = go.Layout(title=r'Bio-Savart Law',
                      scene=dict(xaxis_title=r'x',
                                yaxis_title=r'y',
                                zaxis_title=r'z',
                                aspectratio=dict(x=1, y=1, z=1),
                                camera_eye=dict(x=1.2, y=1.2, z=1.2)))

fig = go.Figure(data = data, layout=layout)
fig.add_scatter3d(x=lx, y=ly, z=lz, mode='lines',
                  line = dict(color='green', width=10))

plot(fig, auto_open=True, filename=r'plotly/bio-savart.html')




