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
from scipy.interpolate import Rbf, griddata
import seaborn as sns
import pandas as pd
# from mayavi import mlab
np.set_printoptions(suppress=True)
plot_width = 16


def select_random_snapsots(U_u, U_v, U_p, t, snapshots_per_dataset):
    n_nodes, n_datasets, snapshots_per_dataset_all = U_u.shape
    U_u_resized = np.empty((n_nodes, n_datasets, snapshots_per_dataset))
    U_v_resized = np.empty((n_nodes, n_datasets, snapshots_per_dataset))
    U_p_resized = np.empty((n_nodes, n_datasets, snapshots_per_dataset))
    t_resized = np.empty((n_datasets, snapshots_per_dataset))
    for i in range(n_datasets):
        N = snapshots_per_dataset_all
        inds = np.sort(np.random.randint(low=0, high=N,
                                         size=snapshots_per_dataset))
        U_u_resized[:, i, :] = U_u[:, i, inds]
        U_v_resized[:, i, :] = U_v[:, i, inds]
        U_p_resized[:, i, :] = U_p[:, i, inds]
        t_resized[i, :] = t[i, inds]
    return U_u_resized, U_v_resized, U_p_resized, t_resized


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
    # c_ticks = np.linspace(np.floor(pmin*100)/100, np.ceil(pmax*100)/100, num=7,
    #                       endpoint=True)
    # norm = mpl.colors.Normalize(vmin=pmin, vmax=pmax)
    lvls = np.linspace(pmin, pmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, p, levels=lvls, cmap=cmap)
    cbar2 = plt.colorbar(cp2, ax=ax2, ticks=c_ticks)
    cbar2.set_label('pressure')
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_eigenfaces(U, x, y, tri):
    for i in range(3):
        u_eig = U[0][:, i].numpy()
        v_eig = U[1][:, i].numpy()
        p_eig = U[2][:, i].numpy()
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

    U_u = np.zeros((n_nodes, n_datasets, snapshots_per_dataset))
    U_v = np.zeros((n_nodes, n_datasets, snapshots_per_dataset))
    U_p = np.zeros((n_nodes, n_datasets, snapshots_per_dataset))
    t = np.zeros((n_datasets, snapshots_per_dataset))
    mu = np.zeros((n_datasets))
    for i, file in enumerate(files):
        data = pickle.load(open(path+file, "rb"))
        N = len(data["time"])
        inds = np.sort(np.random.randint(0, N, size=snapshots_per_dataset))
        U_u[:, i, :] = data["u"][:, inds]
        U_v[:, i, :] = data["v"][:, inds]
        U_p[:, i, :] = data["p"][:, inds]
        t[i] = data["time"][inds]-data["time"][0]
        mu[i] = data["mu"]
        print(file, ":\n", len(data["time"]), len(t[i]),
              data["Re"], data["rho"], data["mu"])
    return U_u, U_v, U_p, t, mu, data["x"], data["y"], data["tri"]


def create_ROM(U_u, U_v, U_p):
    # POD
    S = 3*[None]
    U = 3*[None]
    V = 3*[None]
    mean_tensor = 3*[None]
    for i, tensor in enumerate([U_u, U_v, U_p]):
        n_nodes, n_datasets, n_snapshots_per_dataset = tensor.shape
        tensor.shape = n_nodes, n_datasets*n_snapshots_per_dataset

        # TODO: tensor now shaped (n_nodes, n_datasets, n_snapshots)
        tic = timeit.default_timer()
        # tensor = tensor[:, indices]
        # mean_tensor = tf.reduce_mean(tensor, axis=1, keepdims=True)
        mean_tensor[i] = tf.reduce_mean(tensor, axis=1, keepdims=True)

        mean_centered_data = tf.subtract(tensor, mean_tensor[i])
        S[i], U[i], V[i] = tf.linalg.svd(mean_centered_data, full_matrices=False)
        # Full: S.shape = (T,), u.shape = (n, n), v.shape = (t, t) / comp. time: 6 - 10 s for T=150 / comp. time: 23 - 30 s for T=1500
        # Economy: S.shape = (T,), u.shape = (n, t), v.shape = (t, t) / comp. time: .4 - 1 s for T=150
        # _u2_, _s2_, _v2_ = np.linalg.svd(mean_centered_data, full_matrices=True)
        toc = timeit.default_timer()
        print(S[i].shape, U[i].shape, V[i].shape, toc-tic)
        tensor.shape = n_nodes, n_datasets, n_snapshots_per_dataset
    return S, U, V, mean_tensor


def cylinderwall(x, y):
    # bbx_x = (.1499 < x[0]) & (x[0] < .2501)
    # bbx_y = (.1499 < x[1]) & (x[1] < .2501)
    # return bbx_x and bbx_y and on_boundary
    in_circle = ((x-.2)*(x-.2) + (y-.2)*(y-.2)) < 0.002500001
    return (in_circle)


def plot_mode_amplitude(S, V, t, mu):
    lbl = ["velocity in x direction", "velocity in y direction", "pressure"]
    n = 25
    fig, axs = plt.subplots(2, 1, sharex=True,
                            figsize=(plot_width/2.54, 10/2.54))
    for i in range(3):
        axs[0].plot(np.arange(n), S[i][:n], ".", label=lbl[i])
        cum_en = np.cumsum(S[i])[:n]/np.sum(S[i])*100
        axs[1].plot(np.arange(n), cum_en, ".", label=lbl[i])
    axs[0].legend()
    axs[1].legend()
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
    # for i in range(3):
    if True:
        i = 1
        ax.plot(t.ravel(), V[i][:, 0], linestyle='',
                marker='.', label=lbl[i])
    plt.title("Amplitude of the first mode V over time")
    ax.set_xlabel("time")
    ax.set_ylabel("Modes amplitude")
    ax.legend()
    plt.show()
    return


def my_cos(x, period, amplitude, phase_shift, offset):
    # print(period, amplitude, phase_shift, offset)
    return np.cos((x-phase_shift)*2*np.pi / period) * amplitude + offset


def normalize_phase(V, t, mu):
    """phase of the koefficient of the first mode."""
    n_nodes, n_datasets, snapshots_per_dataset = U_u.shape
    # n_datasets = n_snapshots // snapshots_per_dataset
    V_uy = np.array([V[1][:, 0]]).reshape(n_datasets, snapshots_per_dataset)
    time = t.reshape(n_datasets, snapshots_per_dataset)
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
        # fit[0][2] = 0.
        period[i] = fit[0][0]
        amplitude[i] = fit[0][1]
        phase[i] = fit[0][2]
        time_corrected[i] = time[i]-fit[0][2]  # np.mod(time[i]-fit[0][2], period[i])
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
    ax2.plot(Re[[0, 1]], amplitude[[0, 1]], "ro", label="outlier (no oscillation)")
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
    ax2.plot(.1/Re[[0, 1]], amplitude[[0, 1]], "ro", label="outlier (no oscillation)")
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
    # to allow for a trigonometric interpolation, the oscillations are repeated along the time axis
    # n: number of nodes
    # m: number of snapshots = n_snapshots = n_datasets * snapshots_per_dataset
    # r: truncation rank
    # U_x snapshot matrix. SVD: U = U*S*V

    n_modes = 200
    path = "C:/Users/florianma/Documents/data/flow_around_cylinder/"

    plt.close("all")
    U_u_all, U_v_all, U_p_all, _t_all_, mu, x, y, tri = load_snapshots(path, 800)
    U_u, U_v, U_p, _t_ = select_random_snapsots(U_u_all, U_v_all, U_p_all,
                                                _t_all_, 150)
    S_full, U_full, V_full, M_full = create_ROM(U_u, U_v, U_p)
    phase, period = normalize_phase(V_full, _t_, mu)
    t_all = _t_all_ - phase[:, None]
    t = _t_ - phase[:, None]

    u_m = np.mean(np.mean(U_u, axis=1), axis=1)
    v_m = np.mean(np.mean(U_v, axis=1), axis=1)
    p_m = np.mean(np.mean(U_p, axis=1), axis=1)
    fig, ax = plot_up(u_m, v_m, p_m, x, y, tri)
    plt.show()
    plot_eigenfaces(U_full, x, y, tri)
    plot_mode_amplitude(S_full, V_full, t, mu)
    # asd

    n_nodes, n_datasets, n_snapshots_per_dataset = U_u.shape
    trainingset = np.empty((n_datasets,), dtype=np.bool)
    snapshots_per_dataset = 50  # random choice!
    n_ss = [3, 5, 10, 20, 50, 100, 200]
    mse = np.zeros((3, len(n_ss), n_datasets))
    x_bp = np.zeros((3, len(n_ss), n_datasets), dtype=np.int32)
    set_nr = np.zeros((3, len(n_ss), n_datasets), dtype=np.int32)
    for ns, snapshots_per_dataset in enumerate(n_ss):
        n_modes = snapshots_per_dataset
        U_u, U_v, U_p, t = select_random_snapsots(U_u_all, U_v_all, U_p_all,
                                                  t_all, snapshots_per_dataset)
        snapshots = [U_u, U_v, U_p]
        n_testset = (n_datasets-1)*snapshots_per_dataset
        x1 = mu[:, None] * np.ones((n_datasets, 3*snapshots_per_dataset))  # 9, 3*150
        x2 = np.c_[t-period[:, None], t, t+period[:, None]]  # 9, 3*150
        for i in range(1, n_datasets-1):
            s, e = i, (i+1)
            trainingset[:] = True
            trainingset[s:e] = False
            validationset = ~trainingset
            # need to copy to make array contiguous
            S, U, V, M = create_ROM(U_u[:, trainingset].copy(),
                                    U_v[:, trainingset].copy(),
                                    U_p[:, trainingset].copy())
            x1_train = x1[trainingset, :]
            x2_train = x2[trainingset, :]  # 8, 3*150
            interpolators = 3 * [n_modes * [None]]
            x1_validation = mu[i] * np.ones((snapshots_per_dataset))
            x2_validation = t[i, :]

            # fig, (ax1) = plt.subplots(1, sharex=True,
            #                           figsize=(plot_width/2.54, 10/2.54))
            # ax1.plot(.1/x1_train.ravel(), x2_train.ravel(), "o", color="tab:orange", label="training data")
            # ax1.plot(.1/x1_validation.ravel(), x2_validation.ravel(), "o", color="tab:blue", label="test data")
            # ax1.set_ylabel("time")
            # ax1.set_xlabel("Re")
            # plt.suptitle("Parameter space")
            # plt.legend()
            # plt.show()
            # V_interpolated = 3*[np.zeros((snapshots_per_dataset, n_modes))]
            V_interpolated = [np.zeros((snapshots_per_dataset, n_modes)),
                              np.zeros((snapshots_per_dataset, n_modes)),
                              np.zeros((snapshots_per_dataset, n_modes))]
            n_trainingsets = trainingset.sum()
            res = 3*[np.zeros((n_nodes, snapshots_per_dataset))]
            for j in range(3):
                for k in range(n_modes):
                    d_ = V[j].numpy()[:, k].reshape(n_trainingsets, snapshots_per_dataset).copy()
                    d = np.c_[d_, d_, d_]  # 8, 3*150 repeats in each oscillation
                    # radial basis function interpolator instance
                    # rbfi = Rbf(x1_train.ravel(), x2_train.ravel(), d.ravel())
                    # interpolators[j][k] = rbfi
                    points = np.c_[x1_train.ravel(), x2_train.ravel()]
                    di = griddata(points, d.ravel(),
                                  (x1_validation, x2_validation), method='linear')
                    # rbfi = interpolators[j][k]
                    # di = rbfi(x1_validation, x2_validation)
                    V_interpolated[j][:, k] = di.copy()
                U_hat = U[j].numpy()[:, :n_modes]  # (n, r)
                S_hat = S[j].numpy()[:n_modes]  # (r, r) / (r,) wo  non zero elements
                V_hat = V_interpolated[j]  # (r, m)
                res[j] = np.dot(U_hat, S_hat[:, None]*V_hat.T) + M[j].numpy()  # (n, m)
                mse[j, ns, i] = ((snapshots[j][:, i, :] - res[j])**2).max()
                x_bp[j, ns, i] = snapshots_per_dataset
                set_nr[j, ns, i] = j
                print(i, j, mse[j, ns, i])

            if mu[i] == .002 and snapshots_per_dataset == 100:
                u_int = res[0]
                v_int = res[1]
                p_int = res[2]
                u_orig = U_u[:, i, :]
                v_orig = U_v[:, i, :]
                p_orig = U_p[:, i, :]
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
                               '# snapshots in training set (per simulation)',
                               'data set #'],
                      data=np.c_[mse[:, :, 1:-1].ravel(),
                                 x_bp[:, :, 1:-1].ravel(),
                                 set_nr[:, :, 1:-1].ravel()])

    fig, ax = plt.subplots(figsize=(plot_width/2.54, 10/2.54))
    sns.boxplot(x='# snapshots in training set (per simulation)',
                y='accuracy (mse)', hue='data set #', data=df)
    sns.stripplot(x='# snapshots in training set (per simulation)',
                  y='accuracy (mse)', hue='data set #', data=df,
                  jitter=False, dodge=True, color='black', label='_nolegend')
    # Fix the legend
    handles, labels = ax.get_legend_handles_labels()
    L = ax.legend(handles[0:3], labels[0:3])
    L.get_texts()[0].set_text('velocity (x-component)')
    L.get_texts()[1].set_text('velocity (y-component)')
    L.get_texts()[2].set_text('pressure')
    plt.suptitle("Accuracy of the ROM (cross-validation)")


# old stuff
    #         # V_hat1 = V[j].numpy()[:, :n_modes]  # (r, m)

    #         U_red = U_full[j].numpy()[:, :n_modes]
    #         S_red = S_full[j].numpy()[:n_modes]
    #         V_red = V_full[j].numpy()[:, :n_modes]


    #         print("V interpolated the same as in V?") # nope
    #         for tmp_i in range(50):
    #             print(np.allclose(V_interpolated[j][tmp_i, :], V[j].numpy()[250+tmp_i, :n_modes], atol=1e-6), end=", ")
    #         print()

    #         # print(V_hat.T[:, 0]*S_hat)
    #         # print(V_hat2.T[:, 300]*S_hat)

    #         res0[j] = np.dot(U_red, S_red[:, None]*V_red.T) + M_full[j].numpy()
    #         res0[j].shape = n_nodes, n_datasets, n_snapshots_per_dataset

    #         res0[j][:, i, :]-res[j]

    #         print(np.allclose(U_red, U_hat))
    #         print(np.allclose(S_red, S_hat))
    #         # print(np.allclose(V_red, V_hat1))  # yes
    #         # print(np.allclose(V_red[250:300], V_hat))  # no

    #         # print(np.allclose(V_red.T[:, 250], V_hat.T[:, 0], atol=1e-6))  # no
    #         # print(np.allclose(V_red.T[:, 265], V_hat.T[:, 15], atol=1e-6))  # no
    #         # print(np.allclose(M[j].numpy(), M_full[j].numpy()))
    #         # print()
    #         print(np.allclose(res0[j][:, 5, :], res[j], atol=1e-6))  # no
    #         print()
    # # test: testset i_orig = 5
    # for ts in [0, 15]:
    #     i_orig = 5
    #     res0[1].shape = n_nodes, n_datasets, n_snapshots_per_dataset
    #     res0[2].shape = n_nodes, n_datasets, n_snapshots_per_dataset
    #     for j in range(3):
    #         print(np.allclose(res0[j][:, i_orig, :], res[j], atol=1e-6))
    #         print(np.allclose(res0[j][:, i_orig, ts], res[j][:, ts], atol=1e-6))

    #     u_int = res0[0][:, i_orig, ts]
    #     v_int = res0[1][:, i_orig, ts]
    #     p_int = res0[2][:, i_orig, ts]
    #     u_int2 = res[0][:, ts]
    #     v_int2 = res[1][:, ts]
    #     p_int2 = res[2][:, ts]
    #     np.allclose(u_int, u_int2)

    #     fig, ax = plot_up(u_int, v_int, p_int, x, y, tri)
    #     plt.title("ROM containing test data")
    #     plt.show()


    #     # plt.plot(S[i])



    # # TODO: define t0 according to
    # # get_data()
    # # for root, dirs, files in os.walk("C:/Users/florianma/Documents/data/flow_around_cylinder/"):
    # #     for filename in files:
    #         print(filename)

    # tf_u_approx = tf.add(tf.matmul(U[0], tf.matmul(tf.linalg.diag(S[0]), V[0], adjoint_b=True)), mean_tensor[0]).numpy()
    # tf_v_approx = tf.add(tf.matmul(U[1], tf.matmul(tf.linalg.diag(S[1]), V[1], adjoint_b=True)), mean_tensor[1]).numpy()
    # tf_p_approx = tf.add(tf.matmul(U[2], tf.matmul(tf.linalg.diag(S[2]), V[2], adjoint_b=True)), mean_tensor[2]).numpy()
    # np_u_approx = np.dot(U[0].numpy(), np.dot(np.diag(S[0].numpy()),  V[0].numpy().T))+ mean_tensor[0].numpy()
    # np_v_approx = np.dot(U[1].numpy(), np.dot(np.diag(S[1].numpy()),  V[1].numpy().T))+ mean_tensor[1].numpy()
    # np_p_approx = np.dot(U[2].numpy(), np.dot(np.diag(S[2].numpy()),  V[2].numpy().T))+ mean_tensor[2].numpy()
    # np_u_approx2 = np.dot(U[0].numpy(), S[0].numpy()[:, None]*V[0].numpy().T) + mean_tensor[0].numpy()
    # np_v_approx2 = np.dot(U[1].numpy(), S[1].numpy()[:, None]*V[1].numpy().T) + mean_tensor[1].numpy()
    # np_p_approx2 = np.dot(U[2].numpy(), S[2].numpy()[:, None]*V[2].numpy().T) + mean_tensor[2].numpy()
    # print(np.allclose(np_u_approx, np_u_approx2))
    # print(np.allclose(np_v_approx, np_v_approx2))
    # print(np.allclose(np_p_approx, np_p_approx2))
    # _t_ = 321
    # fig, ax = plot_up(tf_u_approx[:, _t_], tf_v_approx[:, _t_], tf_p_approx[:, _t_], x, y, tri)
    # plt.show()
    # fig, ax = plot_up(np_u_approx2[:, _t_], np_v_approx2[:, _t_], np_p_approx2[:, _t_], x, y, tri)
    # plt.show()