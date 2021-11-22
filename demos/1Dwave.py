#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:04:11 2021

@author: florianma
"""
import numpy as np
from numpy import sin, pi
from scipy.special import jv
import matplotlib.pyplot as plt
from ROM.snapshot_manager import Data
from low_rank_model_construction.basis_function_interpolation import interpolateV, RightSingularValueInterpolator
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close("all")
cmap = matplotlib.cm.get_cmap('jet')
# cmap = matplotlib.cm.get_cmap('parula')
cmap = matplotlib.cm.get_cmap('viridis')
plot_width = 12


m, n = 100, 1001
n2 = 100+1
X = np.empty((m, n))
Xe_fine = np.empty((m, n2))
x = np.linspace(0, 1, m)
xi = np.linspace(0, 1, n2).reshape(-1, 1)
times = np.linspace(0, 1, n)
times[-1] = 1

def front1D(x, time):
    print(time)
    # c = .1# /2
    y = np.zeros_like(x)
    y[(time<x)] = 1.0
    return y

def sigmoid1D(x, time):
    y = 1.0 / (1+np.e**((x-time)*50))
    return y

def plug_1D(x, time):
    print(time)
    c = .1# /2
    y = np.zeros_like(x)
    y[((time-c)<x) & (x<(time+c))] = 1.0
    return y

def random_1D(x, time):
    print(time)
    c = .1# /2
    y = np.zeros_like(x)
    wave = [0.51538182, 0.36639445, 0.8030119 , 0.40366875, 0.23448425,
           0.380502  , 0.01432753, 0.53730148, 0.46815127, 0.99682601,
           0.47923601, 0.68289249, 0.28251217, 0.1522982 , 0.50534002,
           0.00396807, 0.81103336, 0.17195926, 0.19020709, 0.13527934]
    s = len(y[((time-c)<x) & (x<(time+c))])
    if time<.5:
        y[((time-c)<x) & (x<(time+c))] = wave[20-s:]
    else:
        y[((time-c)<x) & (x<(time+c))] = wave[:s]
    return y

def gauss_1D(x, time):
    a = 1 #/2
    b = time
    c = .1 /2
    y = a*np.e**(-(x-b)**2/(2*c**2))
    return y


my_func = sigmoid1D
# fig, ax = plt.subplots()
# ax.plot(x, plug_1D(x, .5))
# ax.plot(x, gauss1D(x, .5))
# plt.show()


for j, time in enumerate(times):
    X[:, j] = my_func(x, time)
for j, time in enumerate(xi[:, 0]):
    Xe_fine[:, j] = my_func(x, time)

grid = [times]
my_data = Data(X, grid)
my_data.decompose(eps=1.0)#-1e-6)
print(my_data.S)
# X_ = my_data.from_reduced_space(my_data.to_reduced_space(X))
# # error = my_data.std_rb(X)
U = my_data.U
S = my_data.S
VT = my_data.VT
r = len(S)

intpld = RightSingularValueInterpolator(grid, VT, method="cubic")
VT_approx_fine = intpld(xi)
X_approx_fine = my_data.from_reduced_space(VT_approx_fine)
VTe_fine = my_data.to_reduced_space(Xe_fine)
Xe_proj_fine = my_data.from_reduced_space(VTe_fine)
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

# plot X: solutions in separate plots
fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
for j, time in enumerate(times):
    # if time in [0., .01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        ax.plot(x, X[:, j], color=cmap(j/len(times)), marker=".", label="FEM solution")
plt.xlabel("x")
plt.ylabel("u(x)")
# plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
for j, time in enumerate(xi[:, 0]):
    if time in [0., .01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        ax.plot(x, X_approx_fine[:, j], color=cmap(j/n2), marker=".", label="FEM solution")
plt.xlabel("x")
plt.ylabel("u(x)")
# plt.legend()
plt.show()


# plot S
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54), sharex=True)
ax1.plot(np.arange(0, r), S, "bo")
ax2.plot(np.arange(0, r), np.cumsum(S)/S.sum(), "bo")
# ax.plot(x[0], Xe[:, j], "g.", label="exact solution")
ax1.set_xlabel("r")
ax1.set_ylabel("singular values")
ax2.set_xlabel("r")
ax1.set_yscale('log')
ax2.set_ylabel("Cutimelative Energy")
# plt.title("dacay of singular values")
ax1.set_xlim([0, r])
plt.show()


# plot VT
for j in range(r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54))
    ax1.plot(x, U[:, j], "b.", label="basis vector")
    ax2.plot(xi[:, 0], VTe_fine[j], "g.")
    ax2.plot(xi[:, 0], VT_approx_fine[j], "r.")
    ax2.plot(times, VT[j], "bx")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax1.set_title("basis vector {:.0f}, energy = {:.1f} %".format(j, S[j]/np.sum(S)*100))
    ax2.set_xlabel("time")
    ax2.set_ylabel("VT(time)")
    ax2.set_title("dynamics of mode {:.0f}".format(j))
    plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = X_approx_fine.shape
w, h = (times.max() - times.min())/ncols, 1.0/nrows
left, right, bottom, top = times.min()-w/2, times.max()+w/2, 0-h/2, 1+h/2
im1 = ax1.imshow(Xe_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im2 = ax2.imshow(Xe_proj_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im3 = ax3.imshow(X_approx_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))

for i in range(len(times)):
    color = X[:, i]
    ax3.scatter(np.ones_like(x) * times[i], x, s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")

cax1 = make_axes_locatable(ax1).append_axes('bottom', size='5%', pad=.5)
cax2 = make_axes_locatable(ax2).append_axes('bottom', size='5%', pad=.5)
cax3 = make_axes_locatable(ax3).append_axes('bottom', size='5%', pad=.5)
fig.colorbar(im1, cax=cax1, orientation="horizontal")
fig.colorbar(im2, cax=cax2, orientation="horizontal")
fig.colorbar(im3, cax=cax3, orientation="horizontal")
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_xlabel("time")
ax2.set_xlabel("time")
ax3.set_xlabel("time")
ax1.set_ylabel("x")
ax2.set_ylabel("x")
ax3.set_ylabel("x")
plt.suptitle("The ROM is based on 10 FEM simulations (o) and can approximate solutions for any time")

ax1.set_title("Exact solution")
ax2.set_title("exact solution projected into reduced space and back")
ax3.set_title("interpolated solution")
plt.show()
