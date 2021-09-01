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
# TODO: move to paraview
# change plotting routine args


def select_random_snapsots(U, xi, dimensions, d1_new):
    [[s1, s2], [d1, d2]] = dimensions
    n, m = s1*s2, d1*d2
    D = len(dimensions[1])
    U.shape = (n, d1, d2)
    xi.shape = (d1, d2, D)
    U_resized = np.empty((n, d1_new, d2))
    xi_resized = np.empty((d1_new, d2, D))
    for i in range(d2):
        inds = np.sort(np.random.randint(low=0, high=d1, size=d1_new))
        inds[0], inds[-1] = 0, d1-1  # we need the edges for the interpolation
        U_resized[..., i] = U[:, inds, i]
        xi_resized[:, i, :] = xi[inds, i, :]
    U.shape = (n, m)
    xi.shape = (m, D)
    U_resized.shape = (n, d1_new*d2)
    xi_resized.shape = (d1_new*d2, D)
    return U_resized, xi_resized


def xxxxx(file, snapshots_per_dataset):
    data = pickle.load(open(file, "rb"))
    N = len(data["time"])
    inds = np.sort(np.random.randint(0, N, size=snapshots_per_dataset))
    u, v, p = data["u"][:, inds], data["v"][:, inds], data["p"][:, inds]
    t = data["time"][inds]-data["time"][0]
    print(file, len(data["t"]), len(t), data["Re"], data["rho"], data["mu"])
    return u, v, p, t, data["mu"]


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
    t[t >= (tmax-1e-6)] = (tmax-1e-6)
    print(t.max())

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
# load_snapshots_cylinder, load_snapshots_cavity moved to rom.snapshot_manager


# def create_ROM(X):
#     # POD
#     n, m = X.shape
#     tensor = X.copy()  # need to copy to make sure array is contiguous

#     tic = timeit.default_timer()
#     # mean_tensor = tf.reduce_mean(tensor, axis=1, keepdims=True)
#     # mean_centered_data = tf.subtract(tensor, mean_tensor)
#     S, U, V = tf.linalg.svd(mean_centered_data, full_matrices=False)
#     toc = timeit.default_timer()
#     print(toc-tic)

#     return S, U, V, mean_tensor


def cylinderwall(x, y):
    # bbx_x = (.1499 < x[0]) & (x[0] < .2501)
    # bbx_y = (.1499 < x[1]) & (x[1] < .2501)
    # return bbx_x and bbx_y and on_boundary
    in_circle = ((x-.2)*(x-.2) + (y-.2)*(y-.2)) < 0.002500001
    return (in_circle)


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
    axs[0].set_yscale('log')
    # axs[0].legend()
    # axs[1].legend()
    axs[0].set_title("First n singular values S")
    axs[1].set_title("Cumulative energy [%]")
    axs[0].set_xlim(0, n)
    axs[0].set_ylim(1, 1000)
    # axs[0].set_ylim(0, S[1])
    # axs[0].set_ylim(bottom=0)
    axs[1].set_xlabel("Snapshot number")
    axs[0].set_ylabel("Singular value")
    axs[1].set_ylabel("Energy in %")
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


def my_cos(x, period, amplitude, phase_shift, offset):
    # print(period, amplitude, phase_shift, offset)
    return np.cos((x-phase_shift)*2*np.pi / period) * amplitude + offset


def normalize_phase(V, time, mu):
    """phase of the koefficient of the first mode."""
    n_datasets, snapshots_per_dataset = time.shape
    # n_datasets = n_snapshots // snapshots_per_dataset
    V_uy = np.array([V[:, 0]]).reshape(n_datasets, snapshots_per_dataset)
    period = np.empty((n_datasets, ))
    amplitude = np.empty((n_datasets, ))
    phase = np.empty((n_datasets, ))
    time_corrected = np.zeros_like(time)
    fits = n_datasets*[None]

    for i in range(n_datasets):
        T = .33  # wild guess
        a = (np.max(V_uy[i])-np.min(V_uy[i]))/2
        offset = (np.max(V_uy[i])+np.min(V_uy[i]))/2
        p = time[i][np.argmax(V_uy[i])]
        p0 = [T, a, p, offset]
        fit = curve_fit(my_cos, time[i], V_uy[i], p0=p0)
        fits[i] = fit
        period[i] = fit[0][0]
        amplitude[i] = fit[0][1]
        phase[i] = fit[0][2]
        time_corrected[i] = time[i]-fit[0][2]
        # np.mod(time[i]-fit[0][2], period[i])
    Re = .1 / mu

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                   figsize=(plot_width/2.54, 10/2.54))
    for i in range(n_datasets):
        fit = fits[i]
        ax1.plot(time[i], my_cos(time[i], *fit[0]), "k--", zorder=n_datasets-i)
        ax1.plot(time[i], V_uy[i], marker=".", zorder=n_datasets-i,
                 label="{:.0f}".format(Re[i]))
        fit[0][2] = 0
        x = np.linspace(-.4, .4, 200)
        ax2.plot(x, my_cos(x, *fit[0]), "k--", zorder=n_datasets-i)
        ax2.plot(time_corrected[i], V_uy[i], marker=".", zorder=n_datasets-i)
    ax1.legend(labelspacing=.0, ncol=2, fontsize=8, title="Reynolds number")
    ax2.annotate(r'Re', xy=(0.05, 0.05),
                 xytext=(-50, -50), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc3,rad=.15"),
                 zorder=n_datasets+1)
    ax1.set_title("Coefficient of the first mode")
    ax2.set_title("Correction of the phase shift")
    ax2.set_xlabel("time [s]")
    ax2.set_xlim([-.4, .4])
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(plot_width/2.54, 10/2.54))
    ax1.plot(Re, period, "o")
    ax2.plot(Re, amplitude, "o", label="estimation from fit")
    ax1.plot(Re[[0, 1]], period[[0, 1]], "ro")
    ax2.plot(Re[[0, 1]], amplitude[[0, 1]], "ro",
             label="outlier (no oscillation)")
    plt.legend()
    ax2.set_xlabel("Reynolds number")
    ax1.set_ylabel("period")
    ax2.set_ylabel("amplitude")
    plt.suptitle("Oscillation of the right singular values")
    # plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True,
                                   figsize=(plot_width/2.54, 10/2.54))
    ax1.plot(.1/Re, period, "o")
    ax2.plot(.1/Re, amplitude, "o", label="estimation from fit")
    ax1.plot(.1/Re[[0, 1]], period[[0, 1]], "ro")
    ax2.plot(.1/Re[[0, 1]], amplitude[[0, 1]], "ro",
             label="outlier (no oscillation)")
    ax2.set_xlabel("viscosity")
    # ax2.set_xlabel("Reynolds number")
    ax1.set_ylabel("period")
    ax2.set_ylabel("amplitude")
    plt.suptitle("Oscillation of the right singular values")
    plt.legend()
    plt.show()
    return phase, period


def normalise(X):
    # return X, 0.0, 1.0
    X_min = X.min(axis=1)[:, None]  # n
    X_max = X.max(axis=1)[:, None]  # n
    X_range = X_max - X_min
    X_range[X_range < 1e-6] = 1e-6
    X_n = (X-X_min)/X_range
    return X_n, X_min, X_range


def split_off(set_i, X, xi, dims, phase_length=False):
    [[s1, s2], [d1, d2]] = dims
    X.shape = (s1, s2, d1, d2)
    xi.shape = (d1, d2, 2)

    i_train = np.delete(np.arange(d2), set_i)
    X_train = X[..., i_train].copy()
    X_valid = X[..., set_i].copy()
    xi_train = xi[:, i_train, :].copy()
    xi_valid = xi[:, set_i, :].copy()

    n_p = 1
    if isinstance(phase_length, np.ndarray):
        offset = np.c_[phase_length[i_train],
                       np.zeros_like(phase_length[i_train])]
        xi_train = np.concatenate((xi_train-offset, xi_train, xi_train+offset),
                                  axis=0)
        # X_train = np.concatenate((X_train, X_train, X_train), axis=2)
        n_p = 3  # number of repetitions of each periods
    X.shape = (s1*s2, d1*d2)
    xi.shape = (d1*d2, 2)
    X_train.shape = (s1*s2, d1*(d2-1))
    xi_train.shape = (n_p*d1*(d2-1), 2)
    X_valid.shape = (s1*s2, d1*1)
    xi_valid.shape = (d1*1, 2)
    return X_train, X_valid, xi_train, xi_valid


def interpolateV(points, values, xi):
    """
    Parameters
    ----------
    points : 2-D ndarray of floats with shape (m, D), or length D tuple of 1-D ndarrays with shape (m,).
        Data point coordinates.
    values : ndarray of float or complex, shape (m, r). V Matrix from SVD (not V.T!)
        Data values.
    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.

    Returns
    -------
    V_interpolated : ndarray
        Array of interpolated values.

    n: n_modes = n_nodes*4 (u,v,p,t)
    D: 2 (time and Tamb or mu)
    r: 12 reduced rank
    """
    m, D = points.shape
    m, r = values.shape  # m, n_modes
    d1, D = xi.shape  # snapshots_per_dataset
    d2 = m // d1  # n_trainingsets
    assert m == d1*d2, "?"

    V_interpolated = np.zeros((d1, r))
    for i in range(r):
        # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
        vals = values[:, i].numpy().copy()

        if vals.shape[0] != points.shape[0]:
            # print("repeat each oscillation")
            # print(points.shape, vals.shape, xi.shape)
            d_ = vals.reshape(d1, d2)  # 50, 1
            # print(d_.shape)
            # 8, 3*150 repeats each oscillation
            d = np.concatenate((d_, d_, d_), axis=0)
            # print(d.shape)
            vals = d.ravel()
            # print(np.allclose(vals, d.ravel()))
            # print(2, points.shape, vals.shape, xi.shape)
        V_interpolated[:, i] = griddata(
            points, vals, xi, method='linear').copy()
    return V_interpolated


def ROM(S, U, V, r):
    U_hat = U.numpy()[:, :r]  # (n, r)
    S_hat = S.numpy()[:r]  # (r, r) / (r,) wo  non zero elements
    V_hat = V[:, :r]  # (d1, r)
    X_approx = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
    return X_approx


def initialROM(X_all, _xi_all_, dims_all):
    X, xi = select_random_snapsots(X_all, _xi_all_, dims_all, 75)
    X_n, X_min, X_range = normalise(X)
    S_full, U_full, V_full = tf.linalg.svd(X_n, full_matrices=False)

    fig, ax = plot_snapshot(X_all[:, 0], x, y, tri)
    plt.show()

    plot_eigenfaces(U_full, x, y, tri)
    plot_mode_amplitude(S_full, V_full, xi)
    return


def ROM_accuracy_animation(X_all, _xi_all_, dims_all):
    X, xi = select_random_snapsots(X_all, _xi_all_, dims_all, 75)
    X_n, X_min, X_range = normalise(X)
    S_full, U_full, V_full = tf.linalg.svd(X_n, full_matrices=False)
    for r in range(1, 50):
        t = 19
        nset = 8
        j = t*dims_all[1][1] + nset
        X_rom_n = ROM(S_full, U_full, V_full.numpy(), r)
        X_rom = X_rom_n*X_range + X_min
        X_diff = np.abs(X_rom_n - X_n)
        rmse = X_diff.mean()
        # fig, ax = plot_snapshot(X_rom[:, j], x, y, tri, tmin=600, tmax=670.0)
        fig, ax = plot_snapshot(X_rom[:, j], x, y, tri, pmin=np.min(X),
                                pmax=np.max(X))
        plt.suptitle("ROM: rank: {:2.0f}, error: {:2.1f}%".format(r, rmse*100))
        path = "C:/Users/florianma/Documents/data/freezing_cavity/"
        plt.savefig(path+"gifs/frames/frame_{:06.0f}.png".format(r))
        plt.close()
    return


if __name__ == "__main__":
    plt.close("all")
    load_snapshots = load_snapshots_cylinder
    path = "C:/Users/florianma/Documents/data/flow_around_cylinder/"
    plot_snapshot = plot_snapshot_cyl
    # load_snapshots = load_snapshots_cavity
    # path = "C:/Users/florianma/Documents/data/freezing_cavity/"
    # plot_snapshot = plot_snapshot_cav
    # to allow for a trigonometric interpolation, the oscillations are repeated
    # along the time axis
    # n: number of nodes (s1) * number of physical quantities (s2)
    # m: number of snapshots = num datasets (d2) * snapshots_per_dataset (d1)
    # r: truncation rank
    # D: dimension parameterspace (2)
    # U_x snapshot matrix. SVD: X = U*S*V
    # X: (n, m) = (s1*s2, d1*d2)
    # xi: (m, D) = (d1*d2, D)

    X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots(path)

    asd
    # ROM_accuracy(X_all, _xi_all_, dims_all)

    print(dims_all)
    [[s1, s2], [d1_all, d2]] = dims_all
    print("n physical quantities", s1)
    print("n_nodes", s2)
    print("snapshots_per_dataset", d1_all, "(will be reduced)")
    print("n_datasets", d2)
    print("n, m", s1*s2, d1_all*d2, X_all.shape)

    initialROM(X_all, _xi_all_, dims_all)

    n_ss = [5, 10, 20, 50, 100, 200]
    # n_ss = [5, 10, 20, 50]
    mse = np.zeros((len(n_ss), d2))
    x_bp = np.zeros((len(n_ss), d2), dtype=np.int32)
    # set_nr = np.zeros((len(n_ss), d2), dtype=np.int32)
    asd
    for ns, d1_new in enumerate(n_ss):
        print(ns, d1_new)
        # ns = 3
        # d1_new = 50
        X, xi = select_random_snapsots(X_all, _xi_all_, dims_all, d1_new)
        dims = [[s1, s2], [d1_new, d2]]
        r = d1_new

        for i in range(1, d2-1):  # we can't extrapolate
            # i = 6
            # set_i, X, xi, dims, phase_length = i, X, xi, dims, phase_length
            X_train, X_valid, xi_train, xi_valid = split_off(
                i, X, xi, dims, phase_length)
            X_n, X_min, X_range = normalise(X_train)

            S, U, V = tf.linalg.svd(X_n, full_matrices=False)
            V_new = interpolateV(points=xi_train, values=V, xi=xi_valid)
            X_rom_n = ROM(S, U, V_new, r)

            X_rom = X_rom_n*X_range + X_min

            if ns == 3 and i == 6:
                for j in [0, d1_new//2]:
                    fig, ax = plot_snapshot(X_valid[:, j], x, y, tri)
                    plt.suptitle("FOM")
                    plt.show()
                    fig, ax = plot_snapshot(X_rom[:, j], x, y, tri)
                    plt.suptitle("ROM")
                    plt.show()

            X_valid_n = (X_valid-X_min) / X_range
            X_diff = np.abs(X_rom_n - X_valid_n)
            # x_min = X_valid_n.min(axis=1)[:, None]
            # x_max = X_valid_n.max(axis=1)[:, None]
            # x_range = x_max-x_min
            # l = (x_range > 1e-6).ravel()
            # X_diff[l, :] = X_diff[l, :] / x_range[l]
            mse[ns, i] = X_diff.mean()

            x_bp[ns, i] = d1_new

    IFE_blue = "#1D2982"
    IFE_red = "#E04A1B"
    df = pd.DataFrame(columns=['accuracy (root mean square deviation)',
                               '# snapshots in each training set'],
                      data=np.c_[mse[:, 1:-1].ravel(),
                                 x_bp[:, 1:-1].ravel()])
    fig, ax = plt.subplots(figsize=(plot_width/2.54, 10/2.54))
    ax1 = sns.boxplot(x='# snapshots in each training set',
                      y='accuracy (root mean square deviation)', data=df)
    sns.stripplot(x='# snapshots in each training set',
                  y='accuracy (root mean square deviation)', data=df,
                  jitter=False, dodge=True, color=IFE_blue, label='_nolegend')
    for i in range(len(ax1.lines)):
        ax1.lines[i].set_linewidth(2)
        ax1.lines[i].set_color(IFE_blue)
    for i in range(6):
        # ax1.artists[i].set_facecolor(IFE_blue)
        ax1.lines[4+i*6].set_color(IFE_red)
    plt.suptitle("Accuracy of the ROM (cross-validation)")
    plt.ylim(0, 0.1)
    ax.yaxis.grid(True)
    plt.show()
