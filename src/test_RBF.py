# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:49:23 2021

@author: florianma
"""
# %%
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap('jet')

path = "/home/fenics/shared/doc/interpolation2/"
P_train_n = np.load(path+"P_train_n.npy")
P_val_n = np.load(path+"P_val_n.npy")
V = np.load(path+"values.npy")

to_be_inetpolated = 6
dT = P_train_n[:, 1]
# %%
indxs = np.arange(10)
test_set = indxs == to_be_inetpolated

for mode in range(10):
    #%%
    Values = V[:, mode].copy()
    P_train_n.shape = (100, 10, 3)
    Values.shape = (100, 10)

    V_orig = Values[:, test_set].copy().ravel()
    xi = P_train_n[:, test_set, :].copy().reshape(100, 3)
    
    x = P_train_n[:, ~test_set, 0].copy().ravel()
    y = P_train_n[:, ~test_set, 1].copy().ravel()
    z = P_train_n[:, ~test_set, 2].copy().ravel()
    vals = Values[:, ~test_set].copy().ravel()

    function = "cubic"
    rbfi = Rbf(x, y, z, vals, function = function, epsilon = 0.1)  # 
                            # 'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                            # 'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                            # 'gaussian': exp(-(r/self.epsilon)**2)
                            # 'linear': r
                            # 'cubic': r**3
                            # 'quintic': r**5
                            # 'thin_plate': r**2 * log(r)
    V_rbf = rbfi(xi[:, 0], xi[:, 1], xi[:, 2])
    # breakpoint()
    V_grid = griddata(P_train_n[:, ~test_set, :].copy().reshape(-1, 3), vals, xi, method = "linear").copy()
    x = P_train_n[:, ~test_set, :]

    fig, ax = plt.subplots()
    for i in range(1, 10):
        if i != to_be_inetpolated:
            ax.plot(P_train_n[:, i, 0], Values[:, i], color = cmap(i/9), marker = ".", linestyle = "", label = "p = {:.2f}".format(P_train_n[0, i, 1]))
        else:
            ax.plot(xi[:, 0], V_orig, color = cmap(to_be_inetpolated/9), marker = "", linestyle = "-", label = "original (p = {:.2f})".format(xi[0, 1]), zorder = 18)
            ax.plot(xi[:, 0], V_grid, "b-", label = "grid interpolated", zorder = 19)
            ax.plot(xi[:, 0], V_rbf, "r-", label = "rbf interpolation", zorder = 20)

    plt.legend()
    plt.title("coefficient of mode {:.0f} over the parameter space".format(mode))
    ax.set_xlabel("time")
    ax.set_ylabel("right singular value")

    plt.xlim(0, 1)
    plt.ylim(Values.min(), Values.max())
    plt.savefig(path+"test_RBF_V_mode{:02.0f}.png".format(mode), dpi = 250)

# %%

# %%
