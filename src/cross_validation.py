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


def select_random_snapsots(U, t, snapshots_per_dataset):
    n, n_datasets, N = U.shape
    U_resized = np.empty((n, n_datasets, snapshots_per_dataset))
    t_resized = np.empty((n_datasets, snapshots_per_dataset))
    for i in range(n_datasets):
        inds = np.sort(np.random.randint(low=0, high=N,
                                         size=snapshots_per_dataset))
        U_resized[:, i, :] = U[:, i, inds]
        t_resized[i, :] = t[i, inds]
    return U_resized, t_resized


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
        u_eig, v_eig, p_eig = np.split(U.numpy()[:, i], 3)
        fig, ax = plot_up(u_eig, v_eig, p_eig, x, y, tri)
        ttl = ("eigen vector #{:.0f}".format(i))
        plt.suptitle(ttl)
        plt.show()
    return


def load_snapshots(path, snapshots_per_dataset):
    files = [f for f in os.listdir(path) if f.endswith(".pickle")]
    data = pickle.load(open(path+files[0], "rb"))

    n_nodes = len(data["x"])
    n_datasets = len(files)

    U = np.zeros((3, n_nodes, n_datasets, snapshots_per_dataset))
    t = np.zeros((n_datasets, snapshots_per_dataset))
    mu = np.zeros((n_datasets))
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


def create_ROM(X):
    # POD
    n, n_datasets, n_snapshots_per_dataset = X.shape
    tensor = X.copy()  # need to copy to make sure array is contiguous
    tensor.shape = n, n_datasets*n_snapshots_per_dataset

    tic = timeit.default_timer()
    mean_tensor = tf.reduce_mean(tensor, axis=1, keepdims=True)
    mean_centered_data = tf.subtract(tensor, mean_tensor)
    S, U, V = tf.linalg.svd(mean_centered_data, full_matrices=False)
    toc = timeit.default_timer()
    print(toc-tic)

    return S, U, V, mean_tensor


def cylinderwall(x, y):
    # bbx_x = (.1499 < x[0]) & (x[0] < .2501)
    # bbx_y = (.1499 < x[1]) & (x[1] < .2501)
    # return bbx_x and bbx_y and on_boundary
    in_circle = ((x-.2)*(x-.2) + (y-.2)*(y-.2)) < 0.002500001
    return (in_circle)


def plot_mode_amplitude(S, V, t, mu):
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


if __name__ == "__main__":
    # arrays are shaped (n_nodes, n_datasets, snapshots_per_dataset)
    # to allow for a trigonometric interpolation, the oscillations are repeated
    # along the time axis
    # n: number of nodes
    # m: number of snapshots = n_snapshots = n_datasets * snapshots_per_dataset
    # r: truncation rank
    # U_x snapshot matrix. SVD: X = U*S*V

    n_modes = 200
    path = "C:/Users/florianma/Documents/data/flow_around_cylinder/"

    plt.close("all")
    X_all, _t_all_, mu, x, y, tri = load_snapshots(path, 800)
    X, _t_ = select_random_snapsots(X_all, _t_all_, 150)
    S_full, U_full, V_full, M_full = create_ROM(X)
    phase, period = normalize_phase(V_full, _t_, mu)
    t_all = _t_all_ - phase[:, None]
    t = _t_ - phase[:, None]

    X_mean = np.mean(np.mean(X, axis=1), axis=1)
    u_m, v_m, p_m = np.split(X_mean, 3)
    fig, ax = plot_up(u_m, v_m, p_m, x, y, tri)
    plt.show()
    plot_eigenfaces(U_full, x, y, tri)
    plot_mode_amplitude(S_full, V_full, t, mu)

    n_datasets = len(t)
    trainingset = np.empty((n_datasets,), dtype=np.bool)
    n_ss = [3, 5, 10, 20, 50]  # , 100, 200]
    mse = np.zeros((len(n_ss), n_datasets))
    x_bp = np.zeros((len(n_ss), n_datasets), dtype=np.int32)
    set_nr = np.zeros((len(n_ss), n_datasets), dtype=np.int32)
    for ns, snapshots_per_dataset in enumerate(n_ss):
        n_modes = snapshots_per_dataset
        X, t = select_random_snapsots(X_all, t_all, snapshots_per_dataset)
        n_testset = (n_datasets-1)*snapshots_per_dataset
        x1 = mu[:, None] * np.ones((n_datasets, 3*snapshots_per_dataset))
        x2 = np.c_[t-period[:, None], t, t+period[:, None]]  # 9, 3*150
        for i in range(1, n_datasets-1):
            s, e = i, (i+1)
            trainingset[:] = True
            trainingset[s:e] = False
            validationset = ~trainingset
            S, U, V, M = create_ROM(X[:, trainingset])
            x1_train = x1[trainingset, :]
            x2_train = x2[trainingset, :]  # 8, 3*150
            x1_validation = mu[i] * np.ones((snapshots_per_dataset))
            x2_validation = t[i, :]

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

            V_interpolated = np.zeros((snapshots_per_dataset, n_modes))
            n_trainingsets = trainingset.sum()
            for k in range(n_modes):
                d_ = V.numpy()[:, k].reshape(n_trainingsets,
                                             snapshots_per_dataset).copy()
                d = np.c_[d_, d_, d_]  # 8, 3*150 repeats each oscillation
                points = np.c_[x1_train.ravel(), x2_train.ravel()]
                di = griddata(points, d.ravel(),
                              (x1_validation, x2_validation), method='linear')
                # rbfi = Rbf(x1_train.ravel(), x2_train.ravel(), d.ravel())
                # di = rbfi(x1_validation, x2_validation)
                V_interpolated[:, k] = di.copy()
            U_hat = U.numpy()[:, :n_modes]  # (n, r)
            S_hat = S.numpy()[:n_modes]  # (r, r) / (r,) wo  non zero elements
            V_hat = V_interpolated  # (r, m)
            res = np.dot(U_hat, S_hat[:, None]*V_hat.T) + M.numpy()  # (n, m)
            mse[ns, i] = ((X[:, i, :] - res)**2).mean()
            x_bp[ns, i] = snapshots_per_dataset
            print(i, mse[ns, i])

        if mu[i] == .002 and snapshots_per_dataset == 100:
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
