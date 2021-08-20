# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:49:23 2021

@author: florianma
"""
from nirom.low_rank_model_construction.basis_function_interpolation import BasisFunctionRegularGridInterpolator as BFRGI
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf, RegularGridInterpolator, interp1d, RBFInterpolator
from scipy.interpolate import griddata, CloughTocher2DInterpolator
import timeit
import pylab
import matplotlib.pyplot as plt
import matplotlib
from pydmd import DMD
LINALG_LIB = "numpy"

cmap = matplotlib.cm.get_cmap('jet')
plt.close("all")
plot_width = 16


path = "/home/fenics/shared/doc/interpolation2/"
path = "C:/Users/florianma/Documents/data/interpolation1/"
P_train_n = np.load(path+"P_train_n.npy")
P_val_n = np.load(path+"P_val_n.npy")
V = np.load(path+"values.npy")

if True:
    # TODO: make 2D
    # var param cant be on regular grid
    P_train_n = P_train_n[..., :2]
    P_val_n = P_val_n[..., :2]


def np_svd(X, full_matrices=False):
    U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    return U, S, Vh


if LINALG_LIB == "tensorflow":
    svd = tf.linalg.svd
    qr = tf.linalg.qr
    transpose = tf.transpose
    matmul = tf.matmul
    reshape = tf.reshape
    inv = tf.linalg.inv
elif LINALG_LIB == "numpy":
    svd = np_svd
    qr = np.linalg.qr
    transpose = np.transpose
    matmul = np.matmul
    reshape = np.reshape
    inv = np.linalg.inv


to_be_inetpolated = 6
dT = P_train_n[:, 1]
indxs = np.arange(10)
test_set = indxs == to_be_inetpolated


class SVDInterpolator:
    def __init__(self, grid, values, method="cubic"):
        self.t = grid[0]
        self.x = grid[1]
        self.method = method
        xgrid, tgrid = np.meshgrid(self.x, self.t)
        self.U, self.S, self.VT = svd(values)
        # print(self.U.shape)
        # print(x)

        # fig, ax = plt.subplots()
        # for i, mode in enumerate(self.U.T):
        #     plt.plot(self.t, mode.real*self.S[i])
        #     plt.title('Modes')
        # plt.show()

        # fig, ax = plt.subplots()
        # for dynamic in self.VT:
        #     plt.plot(self.x, dynamic.real)
        #     plt.title('Dynamics')
        # plt.show()

    def __call__(self, x_new):
        if not isinstance(x_new, np.ndarray):
            x_new = np.array(x_new).reshape(1)
        res = np.zeros((len(self.U), len(x_new)))
        for j in range(len(x_new)):
            dyn = np.zeros_like(self.VT[0])
            for i, dynamic in enumerate(self.VT):
                f = interp1d(self.x, dynamic, kind=self.method)
                dyn[i] = f(x_new[j])
            res[:, j] = matmul(self.U*self.S, dyn)  # n, d1
        return res


def my_square_scatter(axes, x_array, y_array, w, h, **kwargs):
    # size = float(size)
    for x, y in zip(x_array, y_array):
        square = pylab.Rectangle((x-w/2, y-h/2), w, h, **kwargs)
        axes.add_patch(square)
    return True


for mode in range(20):
    Values = V[:, mode].copy()
    P_train_n.shape = (100, 10, 2)
    Values.shape = (100, 10)

    V_orig = Values[:, test_set].copy().ravel()
    xi_val = P_train_n[:, test_set, :].copy().reshape(100, 2)

    # z = P_train_n[:, ~test_set, 2].copy().ravel()


    method1 = "linear"  # "cubic", "nearest"
    function = "cubic"
    # 'multiquadric': sqrt((r/self.epsilon)**2 + 1)
    # 'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
    # 'gaussian': exp(-(r/self.epsilon)**2)
    # 'linear': r
    # 'cubic': r**3
    # 'quintic': r**5
    # 'thin_plate': r**2 * log(r)

# * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # *
    t0 = timeit.default_timer()
    time = P_train_n[:, ~test_set, 0][:, 0].copy()
    x2 = P_train_n[:, ~test_set, 1][0, :].copy()
    grid = (time, x2)
    xy = P_train_n[:, ~test_set, :].copy().reshape(-1, 2)
    x = P_train_n[:, ~test_set, 0].copy().ravel()
    y = P_train_n[:, ~test_set, 1].copy().ravel()
    vals = Values[:, ~test_set].copy()

    t0 = timeit.default_timer()
    # rbfi = Rbf(x, y, vals.ravel(), function=function, epsilon=0.1)
    rbfi = RBFInterpolator(xy, vals.ravel(), kernel=function)
    print("set up RBFInterpolator: {:.4f} s".format(timeit.default_timer()-t0))

    t0 = timeit.default_timer()
    bf_rgi = BFRGI(grid, vals)
    print("set up BFRGI: {:.4f} s".format(timeit.default_timer()-t0))

    t0 = timeit.default_timer()
    rgi = RegularGridInterpolator(grid, vals, method=method1)
    print("set up RegularGridInterpolator: {:.4f} s".format(timeit.default_timer()-t0))

    t0 = timeit.default_timer()
    cti = CloughTocher2DInterpolator(xy, vals.ravel())
    print("set up CloughTocher2DInterpolator: {:.4f} s".format(timeit.default_timer()-t0))

    t0 = timeit.default_timer()
    dmd_lin = SVDInterpolator(grid, vals, method="linear")
    print("set up SVDInterpolator: {:.4f} s".format(timeit.default_timer()-t0))

    t0 = timeit.default_timer()
    dmd_cub = SVDInterpolator(grid, vals, method="cubic")
    print("set up SVDInterpolator: {:.4f} s".format(timeit.default_timer()-t0))





    labels = ["original (p = {:.2f})".format(xi_val[0, 1]),
              'RBF interpolation - ' + function,
              'BF interpolation - ' + "quadratic",
              'regular grid interpolation - '+method1,
              'CloughTocher2DInterpolator',
              'svd linear',
              'svd cubic '
              ]
    data = []
    for i, interpolator in enumerate([rbfi, bf_rgi, rgi, cti, dmd_lin, dmd_cub]):
        t0 = timeit.default_timer()
        if i < 4:
            data[i:i] = [interpolator(xi_val[:, :2])]
        else:
            data[i:i] = [interpolator(xi_val[0, 1]).ravel()]
        print(labels[i+1], timeit.default_timer()-t0)
    data[0:0] = [V_orig]

    # plot interpolation of test data and error
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(plot_width/2.54, plot_width/2.54))
    for d, lbl in zip(data, labels):
        e = V_orig-d
        lbl +="\n(error: {:.4f})".format(np.sum(e**2)**.5)
        ax1.plot(xi_val[:, 0], d,# color=cmap(to_be_inetpolated/9),
                 # marker="", linestyle="dashdot",
                 label=lbl)  # , zorder=18)
        ax2.plot(xi_val[:, 0], e,# color=cmap(to_be_inetpolated/9),
                 # marker="", linestyle="dashdot",
                 label=lbl)  # , zorder=18)
    plt.legend()
    plt.title(
        "error of test interpolation of mode {:.0f} over time".format(mode))
    ax1.set_xlabel("time")
    ax2.set_xlabel("time")
    ax2.set_ylabel("error")
    plt.xlim(0, 1)
    plt.savefig(path+"test_RBF_V_mode{:02.0f}_error.png".format(mode), dpi=250)

    # plot data and interpolation of test data
    fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
    for i in range(1, 10):
        if i != to_be_inetpolated:
            ax.plot(P_train_n[:, i, 0], Values[:, i], color=cmap(i/9),
                    marker=".", linestyle="",
                    label="p = {:.2f}".format(P_train_n[0, i, 1]))
    for d, lbl in zip(data, labels):
        ax.plot(xi_val[:, 0], d, label=lbl)  # , zorder=19)
    plt.legend()
    plt.title(
        "test interpolation of mode {:.0f} over time".format(mode))
    ax.set_xlabel("time")
    ax.set_ylabel("right singular value")
    plt.xlim(0, 1)
    plt.ylim(Values.min(), Values.max())
    plt.savefig(path+"test_RBF_V_mode{:02.0f}.png".format(mode), dpi=250)

    # interpolate on whole domain
    x__ = np.linspace(0, 1, 50)
    XI, YI = np.meshgrid(time, x__, indexing="ij")
    xi = np.concatenate((XI.reshape(-1, 1), YI.reshape(-1, 1)), axis=1)

    data = []
    for i, interpolator in enumerate([rbfi, bf_rgi, rgi, cti, dmd_lin, dmd_cub]):
        t0 = timeit.default_timer()
        if i < 4:
            data[i:i] = [interpolator(xi[:, :2])]
        else:
            data[i:i] = [interpolator(x__)]
        print(timeit.default_timer()-t0)

    lw = .25
    vmin, vmax = data[-1].min(), data[-1].max()

    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(
        3, 2, sharex=True, sharey=True,
        figsize=(plot_width/2.54, 2*plot_width/2.54))
    axs = [ax11, ax21, ax12, ax22, ax31, ax32]
    titles = labels[1:]

    for d, ax, ttl in zip(data, axs, titles):
        ax.pcolor(XI, YI, d.reshape(100, 50), vmin=vmin, vmax=vmax,
                    cmap=cmap, shading='nearest')
        ax.set_title(ttl)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for y_ in x2:
            ax.plot([0, 1], [y_, y_], "k-", lw=lw)
        y_ = xi_val[:, 1]
        ax.plot([0, 1], [y_, y_], "w-", lw=lw)
        ax.set_aspect("equal")

    plt.savefig(path+"test2_RBF_V_mode{:02.0f}.png".format(mode), dpi=500)
    plt.close("all")
