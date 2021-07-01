# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:11:52 2021

@author: florianma
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import tensorflow as tf
import timeit
from scipy.optimize import curve_fit
plot_width = 16.0  # cm


def plot_up(u, v, p, x, y, tri, umin=None, umax=None, pmin=None, pmax=None):
    umin = u.min() if (not umin) else umin
    umax = u.max() if (not umax) else umax
    pmin = p.min() if (not pmin) else pmin
    pmax = p.max() if (not pmax) else pmax
    magnitude = (u**2 + v**2)**.5
    cmap = mpl.cm.inferno

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                   figsize=(plot_width/2.54, 8/2.54))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("velocity")
    ax2.set_title("pressure")

    c_ticks = np.linspace(umin, umax, num=5, endpoint=True)
    norm = mpl.colors.Normalize(vmin=umin, vmax=umax)
    cp1 = ax1.quiver(x, y, u, v, cmap=cmap, color=cmap(norm(magnitude)))
    cbar1 = plt.colorbar(cp1, ax=ax1, ticks=norm(c_ticks))
    cbar1.ax.set_yticklabels(["{:.2f}".format(i) for i in c_ticks])
    cbar1.set_label('velocity')

    c_ticks = np.round(np.linspace(pmin, pmax, num=5, endpoint=True),
                       decimals=2)
    # c_ticks = np.linspace(np.floor(pmin*100)/100, np.ceil(pmax*100)/100, num=7,
    #                       endpoint=True)
    # norm = mpl.colors.Normalize(vmin=pmin, vmax=pmax)
    lvls = np.linspace(pmin, pmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, p, levels=lvls, cmap=cmap)
    cbar2 = plt.colorbar(cp2, ax=ax2, ticks=c_ticks)
    cbar2.set_label('pressure')
    return fig, (ax1, ax2)


if __name__ == "__main__":
    plt.close("all")
    n_frames = 60
    path = "C:/Users/florianma/Documents/data/flow_around_cylinder/"
    fname = "Re060.pickle" # ["Re020.pickle", "Re033.pickle", "Re050.pickle", "Re075.pickle", "Re100.pickle", "Re125.pickle"]
    data = pickle.load(open(path + fname, "rb"))
    i = 0
    for fn in np.linspace(0, len(data["time"]), n_frames, endpoint=False).round(decimals=0).astype(np.int32):
        ts = data["time"][fn]-data["time"][0]
        u, v, p = data["u"][:, fn], data["v"][:, fn], data["p"][:, fn]
        fig, (ax1, ax2) = plot_up(u, v, p, data["x"], data["y"],
                                  data["tri"], -0.7, 2.25, -0.55, 4.3)
        plt.tight_layout()
        # plt.suptitle("t = {:.3f} s".format(ts))
        plt.savefig(path+"gifs/frames/frame_{:06.0f}.png".format(i))
        plt.close()
        i += 1
    umin, umax = np.min(data["u"]), np.max(data["u"])
    pmin, pmax = np.min(data["p"]), np.max(data["p"])
    print(np.min(data["time"]), np.max(data["time"]))
    print(umin, umax)
    print(pmin, pmax)
        # [-0.0781284205759485, 1.9872728360297138, 0.70277024658657470, 4.321829378801773],
        # [-0.1815605410036841, 2.0114381771523426, 0.38725951189669444, 3.681152401635213],
        # [-0.2720953801337758, 2.0183713939933714, 0.20415128746279326, 3.331682474910499],
        # [-0.5606694972964892, 2.1533997987841964, -0.2886356767665060, 3.0701852395475138],
        # [-0.6973726949552412, 2.2355702031939780, -0.5413727608143236, 3.0045801653880893]
