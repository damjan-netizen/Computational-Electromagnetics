# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:44:52 2022

@author: GroH Von Hilfiger
"""

import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
import scipy  as sp
from scipy.integrate import quad
import plotly.graph_objects as go
from IPython.display import HTML
from plotly.offline import plot

## All quantities are dimensionless

#### EXAMPLE 1 ############

# r(t) = <cos(4t),sin(4t),t), charge spread evenly across

# dq = lambda(r')abs(dr/dt)dt; Q=1=int{dq} 
# ==>> lambda = [int{abs(dr/dt)dt}]^-1


t = smp.symbols('t',positive=True)
x, y, z = smp.symbols('x, y, z')

r = smp.Matrix([x,y,z])
r_p = smp.Matrix([smp.cos(4*t), smp.sin(4*t),t])
sep = r - r_p

dr_pdt = smp.diff(r_p,t).norm().simplify()

lam = smp.integrate(dr_pdt, (t, 0, 2*smp.pi))


#now define the integrand for the elecric field

integrand = lam * sep/sep.norm()**3 * dr_pdt

#convert symbolic notation into numerical functions

dExdt = smp.lambdify([t,x,y,z], integrand[0])
dEydt = smp.lambdify([t,x,y,z], integrand[1])
dEzdt = smp.lambdify([t,x,y,z], integrand[2])

def E(x,y,z):
    return np.array([quad(dExdt, 0, 2*np.pi, args=(x,y,z))[0],
                       quad(dEydt, 0, 2*np.pi, args=(x,y,z))[0],
                       quad(dEzdt, 0, 2*np.pi, args=(x,y,z))[0]])

x = np.linspace(-2,2,10)
z = np.linspace(0, 2*np.pi, 10)

xx, yy, zz = np.meshgrid(x,x,z)

#compute the electric field at all points in the meshgrid

E_field = np.vectorize(E, signature='(),(),()->(n)')(xx,yy,zz)
Ex = E_field[:,:,:,0]
Ey = E_field[:,:,:,1]
Ez = E_field[:,:,:,2]

## look at how E_field varies in this meshgrid

plt.hist(Ex.ravel(), bins=100,histtype='step',label='Ex')
plt.hist(Ey.ravel(), bins=100,histtype='step',label='Ey')
plt.hist(Ez.ravel(), bins=100,histtype='step',label='Ez')
plt.legend(loc=0)
plt.xlabel('Electric field magnitude')
plt.ylabel('Frequency')

#since we dont want large arrows on the plot, it's best we make a cutoff point 
#for the Efield magnitude

E_max = 200

Ex[Ex>E_max] = E_max
Ey[Ey>E_max] = E_max
Ez[Ez>E_max] = E_max

Ex[Ex<-E_max] = -E_max
Ey[Ey<-E_max] = -E_max
Ez[Ez<-E_max] = -E_max



tt = np.linspace(0, 2*np.pi, 1000)
lx, ly, lz = np.cos(4*tt), np.sin(4*tt), tt

data = go.Cone(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
               u=Ex.ravel(), v=Ey.ravel(), w=Ez.ravel(),
               colorscale='Inferno', colorbar=dict(title=r'intensity'),
               sizemode="absolute", sizeref=50)

layout = go.Layout(title=r'Electric field around a charged line',
                     scene=dict(xaxis_title=r'x',
                                yaxis_title=r'y',
                                zaxis_title=r'z',
                                aspectratio=dict(x=1, y=1, z=1),
                                camera_eye=dict(x=1.2, y=1.2, z=1.2)))

fig = go.Figure(data = data, layout=layout)
fig.add_scatter3d(x=lx, y=ly, z=lz, mode='lines',
                  line = dict(color='green', width=10))

plot(fig, auto_open=True, filename=r'plotly/Electric field around a charged line.html')