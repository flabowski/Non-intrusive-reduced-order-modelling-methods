#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:49:23 2021

@author: florianma
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.special import xlogy
# from matplotlib import cm
# from mayavi import mlab
import matplotlib
plt.close("all")
cmap = matplotlib.cm.get_cmap('jet')

path = "/home/florianma@ad.ife.no/Documents/_NIROM_/doc/interpolation2/"
path = "C:/Users/florianma/Documents/data/Test_RBF_griddata/"

P_train_n = np.load(path+"P_train_n.npy")
P_val_n = np.load(path+"P_val_n.npy")
V = np.load(path+"values.npy")

to_be_inetpolated = 6
dT = P_train_n[:, 1]

# x, y, z, d = np.random.rand(4, 50)
# d = np.sin(x)
# rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
# xi = yi = zi = np.linspace(0, 1, 20)
# di = rbfi(xi, yi, zi)   # interpolated values

indxs = np.arange(10)
test_set = indxs == to_be_inetpolated
epsilons = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]) / 100
epsilons = np.array([0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                     2.8, 2.85, 2.9, 2.95,
                     3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5]) / 100
# epsilons = np.linspace(.001, .11, 100)
# epsilons = np.array([0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) / 100
# epsilons = [2.85/100]
error = np.zeros((len(epsilons), 10))

function1 = "multiquadric"
function2 = "linear"
method1 = "linear"
method2 = "cubic"
# for mode in range(10):
for mode in [1, 8]:
    # for mode in [8]:
    for i, epsilon in enumerate(epsilons):
        Values = V[:, mode].copy()
        P_train_n.shape = (100, 10, 3)
        Values.shape = (100, 10)

        V_orig = Values[:, test_set].copy().ravel()
        xi = P_train_n[:, test_set, :].copy().reshape(100, 3)

        x = P_train_n[:, ~test_set, 0].copy().ravel()
        y = P_train_n[:, ~test_set, 1].copy().ravel()
        # z = P_train_n[:, ~test_set, 2].copy().ravel()
        vals = Values[:, ~test_set].copy().ravel()

        rbfi1 = Rbf(x, y, vals, function=function1, epsilon=epsilon)  #
        rbfi2 = Rbf(x, y, vals, function=function2, epsilon=epsilon)  #
        # 'multiquadric': sqrt((r/self.epsilon)**2 + 1)
        # 'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
        # 'gaussian': exp(-(r/self.epsilon)**2)
        # 'linear': r
        # 'cubic': r**3
        # 'quintic': r**5
        # 'thin_plate': r**2 * log(r)
        V_rbf = rbfi1(xi[:, 0], xi[:, 1])
        error[i, mode] = np.sum((V_orig-V_rbf)**2)**.5

        if epsilon in np.array([0.05, 2.85, 5.5]) / 100:
            fig, ax = plt.subplots()
            # plt.savefig(path+"test_RBF_V_mode{:02.0f}.png".format(mode), dpi=250)
            ti = np.linspace(0, 1, 100)
            XI, YI = np.meshgrid(ti, ti)
            X = np.concatenate((XI.reshape(-1, 1), YI.reshape(-1, 1)), axis=1)
            # print(XI.shape, YI.shape)
            ZI_rbf = rbfi1(XI, YI)
            ax.pcolor(XI, YI, ZI_rbf, cmap=cmap)
            ax.scatter(x, y, 1, vals, cmap=cmap, edgecolor="k")
            ax.scatter(xi[:, 0], xi[:, 1], 1, xi[:, 1],
                       cmap=cmap, edgecolor="w")
            ax.set_title('RBF interpolation - '+function1 +
                         ", epsilon = {:.4f}".format(epsilon))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("time")
            ax.set_ylabel("temperature")
            plt.show()

        if epsilon == 2.85/100:
            # if epsilon	!= 0.0:
            V_grid = griddata(P_train_n[:, ~test_set, :2].copy().reshape(-1, 2),
                              vals, xi[..., :2], method=method1).copy()
            # x = P_train_n[:, ~test_set, :2]

            fig, ax = plt.subplots()
            for i in range(1, 10):
                if i != to_be_inetpolated:
                    ax.plot(P_train_n[:, i, 0], Values[:, i], color=cmap(i/9),
                            marker=".", linestyle="", label="p={:.2f}".format(P_train_n[0, i, 1]))
                else:
                    ax.plot(xi[:, 0], V_orig, color=cmap(to_be_inetpolated/9),
                            marker="", linestyle="-", label="original (p={:.2f})".format(xi[0, 1]), zorder=18)
                    ax.plot(xi[:, 0], V_grid, "b-",
                            label="grid interpolated", zorder=19)
                    ax.plot(xi[:, 0], V_rbf, "r-",
                            label="rbf interpolation", zorder=20)

            plt.legend()
            plt.title(
                "coefficient of mode {:.0f} over the parameter space".format(mode))
            ax.set_xlabel("time")
            ax.set_ylabel("right singular value")

            plt.xlim(0, 1)
            plt.ylim(Values.min(), Values.max())
            plt.show()
            # print(x, y)

            fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(
                2, 2, sharex=True, sharey=True)
            # plt.savefig(path+"test_RBF_V_mode{:02.0f}.png".format(mode), dpi=250)
            ti = np.linspace(0, 1, 100)
            XI, YI = np.meshgrid(ti, ti)
            X = np.concatenate((XI.reshape(-1, 1), YI.reshape(-1, 1)), axis=1)
            # print(XI.shape, YI.shape)

            ZI_rbf = rbfi1(XI, YI)
            ax11.pcolor(XI, YI, ZI_rbf, cmap=cmap)
            ax11.scatter(x, y, 1, vals, cmap=cmap, edgecolor="k")
            ax11.scatter(xi[:, 0], xi[:, 1], 1, xi[:, 1],
                         cmap=cmap, edgecolor="w")
            ax11.set_title('RBF interpolation - ' + function1)
            ax11.set_xlim(0, 1)
            ax11.set_ylim(0, 1)

            ZI_rbf = rbfi2(XI, YI)
            ax21.pcolor(XI, YI, ZI_rbf, cmap=cmap)
            ax21.scatter(x, y, 1, vals, cmap=cmap, edgecolor="k")
            ax21.scatter(xi[:, 0], xi[:, 1], 1, xi[:, 1],
                         cmap=cmap, edgecolor="w")
            ax21.set_title('RBF interpolation - ' + function2)
            ax21.set_xlim(0, 1)
            ax21.set_ylim(0, 1)

            ZI_grd = griddata(P_train_n[:, ~test_set, :2].copy().reshape(-1, 2),
                              vals, X, method=method1).copy().reshape(100, 100)
            ax12.pcolor(XI, YI, ZI_grd, cmap=cmap)
            ax12.scatter(x, y, 1, vals, cmap=cmap, edgecolor="k")
            ax12.scatter(xi[:, 0], xi[:, 1], 1, xi[:, 1],
                         cmap=cmap, edgecolor="w")
            ax12.set_title('grid interpolation - '+method1)
            ax12.set_xlim(0, 1)
            ax12.set_ylim(0, 1)

            ZI_grd = griddata(P_train_n[:, ~test_set, :2].copy().reshape(-1, 2),
                              vals, X, method=method2).copy().reshape(100, 100)
            ax22.pcolor(XI, YI, ZI_grd, cmap=cmap)
            ax22.scatter(x, y, 1, vals, cmap=cmap, edgecolor="k")
            ax22.scatter(xi[:, 0], xi[:, 1], 1, xi[:, 1],
                         cmap=cmap, edgecolor="w")
            ax22.set_title('grid interpolation - '+method2)
            ax22.set_xlim(0, 1)
            ax22.set_ylim(0, 1)

            # ax1.add_colorbar()
            ax11.set_aspect("equal")
            ax12.set_aspect("equal")
            ax21.set_aspect("equal")
            ax22.set_aspect("equal")
            plt.show()
            # fig, ax = plt.subplots()
            # plt.scatter(x, y, c=vals, marker="s")
            # plt.scatter(xi[:, 0], xi[:, 1], c=V_rbf, marker="s")
            # plt.scatter(xi[:, 0], xi[:, 1], c=V_rbf, marker="s")
            # plt.show()

mean_error = np.mean(error, axis=1)
print(epsilons)
print(mean_error)
fig, ax = plt.subplots()
ax.plot(epsilons, mean_error, "bo")
ax.plot(epsilons, mean_error, "b--")
ax.set_xlabel("epsilon")
ax.set_ylabel("error")
# ax.set_xlim([0, 0.1])
plt.show()


# plt.savefig(path+"error.png".format(mode), dpi=250)
