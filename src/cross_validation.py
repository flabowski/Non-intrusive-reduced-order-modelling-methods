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
    lvls = np.linspace(pmin, pmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, p, levels=lvls, cmap=cmap)
    cbar2 = plt.colorbar(cp2, ax=ax2, ticks=c_ticks)
    cbar2.set_label('pressure')
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_eigenfaces(U, x, y, tri):
    for i in range(3):
        # u_eig, v_eig, p_eig = np.split(U.numpy()[:, i], 3)
        # TODO
        u_eig, v_eig, p_eig, t_eig = np.split(U.numpy()[:, i], 4)
        fig, ax = plot_up(u_eig, v_eig, p_eig, x, y, tri)
        ttl = ("eigen vector #{:.0f}".format(i))
        plt.suptitle(ttl)
        plt.show()
    return


def load_snapshots_cylinder(path, snapshots_per_dataset):
    # FIXME: rename according to convention
    files = [f for f in os.listdir(path) if f.endswith(".pickle")]
    data = pickle.load(open(path+files[0], "rb"))

    n_nodes = len(data["x"])
    n_datasets = len(files)

    U = np.zeros((3, n_nodes, n_datasets, snapshots_per_dataset))
    t = np.zeros((n_datasets, snapshots_per_dataset))
    mu = np.zeros((n_datasets))  # FIXME! make it shaped n_datasets, snapshots_per_dataset
    for i, file in enumerate(files):
        data = pickle.load(open(path+file, "rb"))
        N = len(data["time"])
        inds = np.sort(np.random.randint(0, N, size=snapshots_per_dataset))
        U[0, :, i, :] = data["u"][:, inds]
        U[1, :, i, :] = data["v"][:, inds]
        U[2, :, i, :] = data["p"][:, inds]
        t[i] = data["time"][inds]-data["time"][0]
        mu[i] = data["mu"]
        print(file, ":\n", len(data["time"]), len(t[i]),
              data["Re"], data["rho"], data["mu"])
    U.shape = (3*n_nodes, n_datasets, snapshots_per_dataset)
    return U, t, mu, data["x"], data["y"], data["tri"]


def load_snapshots_cavity(path):
    x = np.load(path+"x.npy")
    y = np.load(path+"y.npy")
    tri = np.load(path+"tri.npy")
    time = np.load(path+"Tamb400_time.npy")
    Ts = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625])
    Ts = np.array([400, 500, 600])
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
    n = 25
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

    fig, ax = plt.subplots()
    ax.plot(t.ravel(), V[:, 0], linestyle='',
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
    print(X.shape)
    X_min = X.min(axis=1)[:, None]  # n
    X_max = X.max(axis=1)[:, None]  # n
    print(X_min.shape, X_max.shape)
    X_range = X_max - X_min
    X_range[X_range < 1e-6] = 1e-6
    X_n = (X-X_min)/X_range
    print(X_n.min(), X_n.max())
    return X_n, X_min, X_range


def split_off(set_i, X, xi, dims):
    [[s1, s2], [d1, d2]] = dims
    X.shape = (s1, s2, d1, d2)
    xi.shape = (d1, d2, 2)

    i_train = np.delete(np.arange(d2), set_i)
    X_train = X[..., i_train].copy()
    X_valid = X[..., set_i].copy()
    xi_train = xi[:, i_train, :].copy()
    xi_valid = xi[:, set_i, :].copy()

    X.shape = (s1*s2, d1*d2)
    X_train.shape = (s1*s2, d1*(d2-1))
    X_valid.shape = (s1*s2, d1*1)
    xi.shape = (d1*d2, 2)
    xi_train.shape = (d1*(d2-1), 2)
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
    print("m, D", m, D)
    m, r = values.shape  # m, n_modes
    print("m, r", m, r)
    d1, D = xi.shape  # snapshots_per_dataset
    print("d1, D", d1, D)
    d2 = m // d1  # n_trainingsets
    assert m == d1*d2, "?"

    V_interpolated = np.zeros((d1, r))
    for i in range(r):
        vals = values[:, i].numpy()  # .copy() ?
        V_interpolated[:, i] = griddata(points, vals, xi, method='linear')  # last 3 entries
    return V_interpolated


def ROM(S, U, V, r):
    U_hat = U.numpy()[:, :r]  # (n, r)
    S_hat = S.numpy()[:r]  # (r, r) / (r,) wo  non zero elements
    V_hat = V[:, :r]  # (d1, r)
    X_approx = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
    return X_approx


if __name__ == "__main__":
    # TODO: normalize data
    # TODO: structure. generalize!
    load_snapshots = load_snapshots_cylinder
    load_snapshots = load_snapshots_cavity
    path = "C:/Users/florianma/Documents/data/freezing_cavity/"
    # arrays are shaped (n_nodes, n_datasets, snapshots_per_dataset)
    # to allow for a trigonometric interpolation, the oscillations are repeated
    # along the time axis
    # n: number of nodes
    # m: number of snapshots = n_snapshots = n_datasets * snapshots_per_dataset
    # r: truncation rank
    # U_x snapshot matrix. SVD: X = U*S*V

    n_modes = 200

    plt.close("all")
    X_all, _xi_all_, x, y, tri, dims_all = load_snapshots(path)
    print(dims_all)
    [[s1, s2], [d1_all, d2]] = dims_all
    d1_new = 75
    print("n physical quantities", s1)
    print("n_nodes", s2)
    print("snapshots_per_dataset", d1_all, "reduced to", d1_new)
    print("n_datasets", d2)
    print("n, m", s1*s2, d1_all*d2, X_all.shape)
    X, xi = select_random_snapsots(X_all, _xi_all_, dims_all, d1_new)
    dims = [[s1, s2], [d1_new, d2]]
    print("n, m", s1*s2, d1_new*d2, X.shape)
    X_n, X_min, X_range = normalise(X)
    S_full, U_full, V_full = tf.linalg.svd(X_n, full_matrices=False)
    # S_full, U_full, V_full, M_full = create_ROM(X)
    # FIXME
    # phase, period = normalize_phase(V_full, _t_, mu)
    # t_all = _t_all_ - phase[:, None]
    # t = _t_ - phase[:, None]
    # t_all = _t_all_
    # t = _t_

    # X_mean = np.mean(np.mean(X, axis=1), axis=1)
    # print(X_mean.shape)
    # FIXME
    # u_m, v_m, p_m, t_m = np.split(X_mean, 4)
    # print(u_m.shape)
    # fig, ax = plot_up(u_m, v_m, p_m, x, y, tri)
    # plt.show()
    u, v, p, t = np.split(X_all[:, 0], 4)
    fig, ax = plot_up(u, v, p, x, y, tri)
    plt.show()
    print(U_full.shape)

    plot_eigenfaces(U_full, x, y, tri)
    plot_mode_amplitude(S_full, V_full, xi)

    # set_i = 0
    # X_train, X_valid, xi_train, xi_valid = split_off(set_i, X_n, xi, dims)
    # asd
    # n_datasets = len(t)
    # trainingset = np.empty((d2,), dtype=np.bool)
    n_ss = [30, 5, 10, 20, 50, 100, 200]
    mse = np.zeros((len(n_ss), d2))
    # x_bp = np.zeros((len(n_ss), d2), dtype=np.int32)
    # set_nr = np.zeros((len(n_ss), d2), dtype=np.int32)
    for ns, d1_new in enumerate(n_ss):
        X, xi = select_random_snapsots(X_all, _xi_all_, dims_all, d1_new)
        dims = [[s1, s2], [d1_new, d2]]
        r = d1_new

        # X, t = select_random_snapsots(X_all, t_all, d1_new)
        # n_testset = (d2-1)*d1_new
        # FIXME: no longer periodic
        # x1 = mu[:, None] * np.ones((n_datasets, 3*d1_new))
        # x2 = np.c_[t-period[:, None], t, t+period[:, None]]  # 9, 3*150
        # x1 = mu[:, None] * np.ones((d2, d1_new))
        # x2 = t  # 9, 150
        for i in range(1, d2-1):  # we can't extrapolate
            X_train, X_valid, xi_train, xi_valid = split_off(i, X, xi, dims)
            X_n, X_min, X_range = normalise(X_train)
            S, U, V = tf.linalg.svd(X_n, full_matrices=False)
            print(xi_train.shape, V.shape, xi_valid.shape)
            V_new = interpolateV(points=xi_train, values=V, xi=xi_valid)
            X_rom = ROM(S, U, V_new, r)

            X_valid_n = (X_valid-X_min) / X_range
            err = ((X_rom - X_valid_n)**2).mean()



            X_scaled = X_rom*X_range + X_min

            for t in range(4):
                u, v, p, t = np.split(X_scaled[:, t], 4)
                fig, ax = plot_up(u, v, p, x, y, tri)
                plt.show()

            # X_valid
            asd
            # U_hat = U.numpy()[:, :r]  # (n, r)
            # S_hat = S.numpy()[:r]  # (r, r) / (r,) wo  non zero elements
            # V_hat = V_interpolated[:, :r]  # (d1, r)
            # X_rom = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
            # X_scaled = X_rom*X_range + X_min




            # s, e = i, (i+1)
            # trainingset[:] = True
            # trainingset[s:e] = False
            # validationset = ~trainingset
            # S, U, V, M = create_ROM(X[:, trainingset])
            # x1_train = x1[trainingset, :]
            # x2_train = x2[trainingset, :]  # 8, 3*150
            # x1_validation = mu[i] * np.ones((d1_new))
            # x2_validation = t[i, :]

            # fig, (ax1) = plt.subplots(1, sharex=True,
            #                           figsize=(plot_width/2.54, 10/2.54))
            # ax1.plot(.1/x1_train.ravel(), x2_train.ravel(), "o",
            #          color="tab:orange", label="training data")
            # ax1.plot(.1/x1_validation.ravel(), x2_validation.ravel(), "o",
            #          color="tab:blue", label="test data")
            # ax1.set_ylabel("time")
            # ax1.set_xlabel("Re")
            # plt.suptitle("Parameter space")
            # plt.legend()
            # plt.show()
            # points = np.c_[x1_train.ravel(), x2_train.ravel()]
            # n_trainingsets = trainingset.sum()
            # for k in range(n_modes):
            #     # FIXME
            #     d_ = V.numpy()[:, k].reshape(n_trainingsets,
            #                                  d1_new).copy()
            #     # d = np.c_[d_, d_, d_]  # 8, 3*150 repeats each oscillation
            #     d = d_
            #     points = np.c_[x1_train.ravel(), x2_train.ravel()]
            #     di = griddata(points, d.ravel(),
            #                   (x1_validation, x2_validation), method='linear')
            #     # rbfi = Rbf(x1_train.ravel(), x2_train.ravel(), d.ravel())
            #     # di = rbfi(x1_validation, x2_validation)
            #     V_interpolated[:, k] = di.copy()
            U_hat = U.numpy()[:, :n_modes]  # (n, r)
            S_hat = S.numpy()[:n_modes]  # (r, r) / (r,) wo  non zero elements
            V_hat = V_interpolated  # (r, m)
            res = np.dot(U_hat, S_hat[:, None]*V_hat.T) + M.numpy()  # (n, m)
            mse[ns, i] = ((X[:, i, :] - res)**2).mean()
            x_bp[ns, i] = d1_new
            print(i, mse[ns, i])

        if mu[i] == .002 and d1_new == 100:
            u_int, v_int, p_int = np.split(res, 3)
            u_orig, v_orig, p_orig = np.split(X[:, i, :], 3)
            for ts in [0, 15]:
                fig, ax = plot_up(u_orig[:, ts], v_orig[:, ts],
                                  p_orig[:, ts], x, y, tri)
                plt.suptitle("FOM")
                plt.show()

                fig, ax = plot_up(u_int[:, ts], v_int[:, ts], p_int[:, ts],
                                  x, y, tri)
                plt.suptitle("ROM")
                plt.show()

    df = pd.DataFrame(columns=['accuracy (mse)',
                               '# snapshots in training set (per simulation)'],
                      data=np.c_[mse[:, 1:-1].ravel(),
                                 x_bp[:, 1:-1].ravel()])

    fig, ax = plt.subplots(figsize=(plot_width/2.54, 10/2.54))
    sns.boxplot(x='# snapshots in training set (per simulation)',
                y='accuracy (mse)', data=df)
    sns.stripplot(x='# snapshots in training set (per simulation)',
                  y='accuracy (mse)', data=df,
                  jitter=False, dodge=True, color='black', label='_nolegend')
    plt.suptitle("Accuracy of the ROM (cross-validation)")
