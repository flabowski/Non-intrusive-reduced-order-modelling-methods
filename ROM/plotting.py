#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:36:11 2021

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plot_width = 16


def plot_snapshot_cyl(snapshot, x, y, tri,
                  umin=None, umax=None, pmin=None, pmax=None):
    u, v, p, t = np.split(snapshot, 4)
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
    lvls = np.linspace(pmin, pmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, p, levels=lvls, cmap=cmap)
    cbar2 = plt.colorbar(cp2, ax=ax2, ticks=c_ticks)
    cbar2.set_label('pressure')
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_snapshot_cav(snapshot, x, y, tri, umin=None, umax=None, tmin=None, tmax=None):
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


def plot_eigenfaces(U, x, y, tri):
    for i in range(3):
        fig, ax = plot_snapshot(U.numpy()[:, i], x, y, tri)
        ttl = ("eigen vector #{:.0f}".format(i))
        plt.suptitle(ttl)
        plt.show()
    return


def plot_mode_amplitude(S, V, xi):
    t = xi[:, 0]
    # lbl = ["velocity in x direction", "velocity in y direction", "pressure"]
    n = 200
    fig, axs = plt.subplots(2, 1, sharex=True,
                            figsize=(plot_width/2.54, 10/2.54))
    # for i in range(3):
    axs[0].plot(np.arange(n), S[:n], ".")
    cum_en = np.cumsum(S)[:n]/np.sum(S)*100
    axs[1].plot(np.arange(n), cum_en, ".")
    # axs[0].legend()
    # axs[1].legend()
    axs[0].set_title("First n singular values S")
    axs[1].set_title("Cumulative energy [%]")
    axs[0].set_xlim(0, n)
    axs[0].set_ylim(0, S[1])
    axs[1].set_xlabel("Snapshot number")
    axs[0].set_ylabel("Singular value")
    axs[1].set_ylabel("Energy in %")
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim([0, 100])
    plt.tight_layout()
    plt.show()

    # for i in range(3):
    #     # mlab.points3d(mu, t, V[i][:, 0]*10)
    #     # mlab.show()
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     # fig, ax = plt.subplots()
    #     ax.plot3D(mu, t, V[i][:, 0]*100, marker="o")
    #     plt.title("Amplitude of the first mode: "+lbl[i])
    #     ax.set_xlabel("viscosity")
    #     ax.set_ylabel("time")
    #     plt.show()

    for i in range(2):
        fig, ax = plt.subplots()
        ax.plot(t.ravel(), V[:, i], linestyle='',
                marker='.')
        plt.title("Amplitude of the first mode V over time")
        ax.set_xlabel("time")
        ax.set_ylabel("Modes amplitude")
        # ax.legend()
        plt.show()
    return