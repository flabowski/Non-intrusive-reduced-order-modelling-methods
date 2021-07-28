#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:05:51 2021

@author: florianma
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from pydmd import DMD
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap('jet')

path = "/home/fenics/shared/doc/interpolation2/"
P_train_n = np.load(path+"P_train_n.npy")
P_val_n = np.load(path+"P_val_n.npy")
V = np.load(path+"values.npy")

r = 25
to_be_inetpolated = 6
# dT = P_train_n[:, 1]

# indxs = np.arange(10)
# test_set = indxs == to_be_inetpolated
print(V.shape)
V_int = V.reshape(100, 10, 1000)
P_train_n.shape = (100, 10, 3)

# dT = P_train_n[:, to_be_inetpolated, 1].copy()
# q = P_train_n[:, to_be_inetpolated, 2].copy()

# X = V_int[:, to_be_inetpolated, :150].copy()  # shape: 100, 1000 = time, "space"
# x = np.arange(0, 150)

X = V_int[:, 1:, 0].copy()  # shape: 100, 1000 = time, "space"
x = np.arange(0, 9)

t = P_train_n[:, to_be_inetpolated, 0].copy()

xgrid, tgrid = np.meshgrid(x, t)

titles = ['$f$']
data = [X]

fig, ax = plt.subplots()
for n, title, d in zip(range(131, 134), titles, data):
    # plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
# plt.show()
plt.savefig(path+"DMDinput.png", dpi=250)
plt.close()

fig, ax = plt.subplots()
for i in range(9):
    # x[i]
    ax.plot(t, X[:, i], color=cmap(i/10), marker=".", linestyle="-", label="p = {:.0f}".format(x[i]))

plt.legend()
plt.title("coefficients of the first mode")
ax.set_xlabel("time")
ax.set_ylabel("right singular value")

plt.xlim(0, 1)
plt.ylim(X[:, :r].min(), X[:, :r].max())
plt.savefig(path+"DMD8Modes.png", dpi=250)
plt.close()

dmd = DMD(svd_rank=r)
dmd.fit(X.T)  # expected shape = (65, 129) = (spacial, time)

for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(np.sqrt(eig.imag**2+eig.real**2) - 1)))

fig, ax = plt.subplots()
dmd.plot_eigs(show_axes=True, show_unit_circle=True)
plt.savefig(path+"DMDeigs.png", dpi=250)
plt.close()


fig, ax = plt.subplots()
for i, mode in enumerate(dmd.modes.T):
    # print(i)
    if i < 10:
        plt.plot(x, mode.real)
        plt.title('Modes')
plt.savefig(path+"DMDModes.png", dpi=250)
plt.close()
# plt.show()

fig, ax = plt.subplots()
for i, dynamic in enumerate(dmd.dynamics):
    print(i)
    plt.plot(t, dynamic.real)
plt.title('Dynamics')
plt.savefig(path+"DMDDynamics1.png", dpi=250)
plt.ylim(-.001, .001)
plt.savefig(path+"DMDDynamics2.png", dpi=250)
plt.close()
# plt.show()

fig = plt.figure(figsize=(17, 6))
# fig, ax = plt.subplots()

for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()

plt.savefig(path+"DMDresult.png", dpi=250)
plt.close()
# plt.show()
