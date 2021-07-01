# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:42:51 2021
@author: florianma
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import timeit
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import seaborn as sns
import pandas as pd
# from mayavi import mlab
np.set_printoptions(suppress=True)
plot_width = 16


def load_snapshots_cavity(path):
    x = np.load(path+"x.npy")
    y = np.load(path+"y.npy")
    tri = np.load(path+"tri.npy")
    time = np.load(path+"Tamb400_time.npy")
    Ts = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625])
    s1 = 4  # u, v, p and T
    s2 = len(x)
    d1 = len(time)
    d2 = len(Ts)
    n, m = s1*s2, d1*d2
    dimensions = [[s1, s2], [d1, d2]]
    D = len(dimensions[1])  # time and wall temperature
    print("n physical quantities", s1)
    print("n_nodes", s2)
    print("snapshots_per_dataset", d1)
    print("n_datasets", d2)
    U = np.zeros((s1, s2, d1, d2))
    xi = np.zeros((d1, d2, D))
    for i, t_amb in enumerate(Ts):  # iteration along d2
        # t_amb = 600
        # path = "C:/Users/florianma/Documents/data/freezing_cavity/"
        uv = np.load(path+"Tamb{:.0f}_velocity.npy".format(t_amb)).T.copy()
        uv.shape = (2, s2, d1)
        u, v = uv
        time = np.load(path+"Tamb{:.0f}_time.npy".format(t_amb))  # d1
        p = np.load(path+"Tamb{:.0f}_pressure.npy".format(t_amb)).T
        temp = np.load(path+"Tamb{:.0f}_temperature.npy".format(t_amb)).T
        U[0, :, :, i] = u
        U[1, :, :, i] = v
        U[2, :, :, i] = p
        U[3, :, :, i] = temp
        xi[:, i, 0] = time
        xi[:, i, 1] = t_amb
        print(t_amb, ":", p.shape, len(time))
    U.shape = (n, m)
    xi.shape = (m, D)
    return U, xi, x, y, tri, dimensions


def plot_up(snapshot, x, y, tri, umin=None, umax=None, tmin=None, tmax=None):
    u, v, p, t = np.split(snapshot, 4)
    u.shape = v.shape = p.shape = t.shape = -1
    umin = u.min() if (not umin) else umin
    umax = u.max() if (not umax) else umax
    tmin = t.min() if (not tmin) else tmin
    tmax = t.max() if (not tmax) else tmax
    magnitude = (u**2 + v**2)**.5
    cmap = mpl.cm.inferno
    t[t>tmax] = tmax

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                                   figsize=(plot_width/2.54, 8/2.54))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("velocity")
    ax2.set_title("temperature")

    c_ticks = np.linspace(umin, umax, num=5, endpoint=True)
    norm = mpl.colors.Normalize(vmin=umin, vmax=umax)
    cp1 = ax1.quiver(x, y, u, v, cmap=cmap, color=cmap(norm(magnitude)))
    cbar1 = plt.colorbar(cp1, ax=ax1, ticks=norm(c_ticks))
    cbar1.ax.set_yticklabels(["{:.2f}".format(i) for i in c_ticks])
    # cbar1.set_label('velocity')

    c_ticks = np.round(np.linspace(tmin, tmax, num=5, endpoint=True),
                       decimals=2)
    lvls = np.linspace(tmin, tmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, t, levels=lvls, cmap=cmap, extend='min')
    cbar2 = plt.colorbar(cp2, ax=ax2, ticks=c_ticks)
    # cbar2.set_label('pressure')
    plt.tight_layout()
    return fig, (ax1, ax2)



if __name__ == "__main__":
    path = "C:/Users/florianma/Documents/data/freezing_cavity/"
    X_all, _xi_all_, x, y, tri, dims_all = load_snapshots_cavity(path)
    X_all.shape = (4, 3015, 5000, 10)
    u, v = X_all[0], X_all[1]
    magnitude = (u**2 + v**2)**.5
    umin = 0.
    umax = 0.007
    tmin = 600
    tmax = 670

    set_ = 9
    i = 0
    for ts in np.linspace(0, 4999, 120).astype(np.int32):
        snapshot = X_all[..., ts, set_]
        plot_up(snapshot, x, y, tri, umin, umax, tmin, tmax)
        plt.savefig(path+"gifs/frames/frame_{:06.0f}.png".format(i))
        plt.close()
        i += 1


