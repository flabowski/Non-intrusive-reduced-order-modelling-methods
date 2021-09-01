#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:30:48 2021

@author: florianma
"""
import sys
from ROM.plotting import plot_snapshot_cav
from ROM.snapshot_manager import get_snapshot_matrix, Data
import matplotlib.pyplot as plt
import numpy as np
# import sys
# import os
# from os.path import isfile, join
import tensorflow as tf
from scipy.interpolate import Rbf
import timeit
from scipy.interpolate import griddata
np.set_printoptions(suppress=True)
plot_width = 16
print()
print(sys.path)
# print(np.__file__)
sys.path.append('/home/fenics/shared/')
print()
print(sys.path)
sys.path


class ROM():
    def __init__(self, data):
        # (s1, s2, d1, d2) = X.shape
        # (d1_, d2_, D) = points.shape
        # print(s1, s2, d1, d2)
        # print(d1_, d2_, D)

        # self.n_state_variables = s1
        # self.n_nodes = s2
        # self.n_time_instances = d1
        # self.n_parameter_values = d2
        # self.n_parameters = D
        # self.N, self.M = s1*s2, d1*d2
        self.reduced_rank = data.M
        # print(X.shape, P.shape)
        # print(self.N, self.M)
        # print(X.shape)

        # X.shape = (self.N, self.M)
        # points.shape = (self.M, D)
        # self.X, self.points = X, points
        # self.X_n, self.X_min, self.X_range = normalise(X, axis=1)
        # self.points_n, self.points_min, self.points_range = normalise(points,
        #                                           axis=0)
        self.data = data
        # (N, R), (R, R), (M, M)
        S, U, V = tf.linalg.svd(data.X_train_n, full_matrices=False)

        path = "/home/fenics/shared/doc/"
        np.save(path+"P_train_n.npy", data.P_train_n)
        np.save(path+"P_val_n.npy", data.P_val_n)
        np.save(path+"values.npy", V)

        self.S = S.numpy()
        self.U = U.numpy()
        self.V = V.numpy()
        # Psi_s = V.numpy()
        # for i in range(10):
        #     for j in range(10):
        #         print(i, j, "{:.6f}".format(np.sum(Psi_s[:, j] * Psi_s[:, i])))
        return

    def interpolateV(self, xi, method):
        """
        Parameters
        ----------
        xi : 2-D ndarray of floats with shape (m, D), or length D tuple of
            ndarrays broadcastable to the same shape.
            Points at which to interpolate data.

        Returns
        -------
        V_interpolated : ndarray
            Array of interpolated values.

        n: n_modes = n_nodes*4 (u, v, p, t)
        D: 3 (time and dT and Q)
        r: 12 reduced rank
        """
        # m, D = points.shape
        # m, r = values.shape  # m, n_modes
        # d1, D = xi.shape  # snapshots_per_dataset
        # d2 = m // d1  # n_trainingsets
        # assert m == d1*d2, "?"
        data = self.data

        V_interpolated = np.zeros((data.n_time_instances, self.reduced_rank))
        print("interpolating... ")
        for i in range(self.reduced_rank):
            print(i, end=", ")
            vals = self.V[:, i].copy()
            if method == "rbf":
                x, y, z, d = (data.P_train_n[:, 0], data.P_train_n[:, 1],
                              data.P_train_n[:, 2], vals)
                # radial basis function interpolator instance
                rbfi = Rbf(x, y, z, d, function="linear")
                V_interpolated[:, i] = rbfi(xi[:, 0], xi[:, 1], xi[:, 2])
            else:
                V_interpolated[:, i] = griddata(
                    data.P_train_n, vals, xi, method=method).copy()
        self.V_interpolated = V_interpolated
        return V_interpolated

    def predict(self, V_):
        r = self.reduced_rank
        U_hat = self.U[:, :r]  # (n, r)
        S_hat = self.S[:r]  # (r, r) / (r,) wo  non zero elements
        V_hat = V_[:, :r]  # (d1, r)
        X_approx_n = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
        X_approx = self.data.rescaleX(X_approx_n)  # n, d1
        return X_approx, X_approx_n


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


if __name__ == "__main__":
    print("main")
    X_all, P_all = get_snapshot_matrix()
    my_data = Data(X_all, P_all, 4)
    my_data.test()

    path = "/home/fenics/shared/doc/cavity_solidification_dt(0)/"
    x, y = np.load(path+"TambLIN_x.npy"), np.load(path+"TambLIN_y.npy")
    tri = np.load(path+"TambLIN_tri.npy")

    my_ROM = ROM(my_data)
    my_ROM.reduced_rank = 100

    V_linear = my_ROM.interpolateV(my_data.P_val_n, method="linear")
    X_pred_lin, X_pred_lin_n = my_ROM.predict(V_linear)
    V_rbf = my_ROM.interpolateV(my_data.P_val_n, method="rbf")
    X_pred_rbf, X_pred_rbf_n = my_ROM.predict(V_rbf)
    i = 50
    plot_snapshot_cav(X_pred_lin[:, i], x, y, tri)
    plt.savefig("./X_pred_lin.png")
    plot_snapshot_cav(X_pred_rbf[:, i], x, y, tri)
    plt.savefig("./X_pred_rbf.png")
    plot_snapshot_cav(my_data.X_val[:, i], x, y, tri)
    plt.savefig("./X_val.png")
    for i in range(len(my_data.X_val[0, :])):
        err_lin = np.sum((my_data.X_val[:, i] - X_pred_lin[:, i])**2)**.5
        err_rbf = np.sum((my_data.X_val[:, i] - X_pred_rbf[:, i])**2)**.5
        print(i, err_lin, err_rbf)

    t = my_data.P_val_n[:, 0]

    fig, (ax) = plt.subplots()
    for i in range(2):
        ax.plot(t, V_linear[:, 0], "g-")
        ax.plot(t, V_linear[:, 1], "g-")
        ax.plot(t, V_linear[:, 2], "g-")
        # ax.plot(t, V_linear[:, 3], "g-")
        # ax.plot(t, V_linear[:, 4], "g-")
        ax.plot(t, V_rbf[:, 0], "r-")
        ax.plot(t, V_rbf[:, 1], "r-")
        ax.plot(t, V_rbf[:, 2], "r-")
        # ax.plot(t, V_rbf[:, 3], "r-")
        # ax.plot(t, V_rbf[:, 4], "r-")
    plt.savefig("./V.png")

    # # plot pred + val
    # print(X_pred)
    # print(X_val)
    # asd
    # print("S")

    # n_state_variables = 4
    # n_nodes = 3015
    # n_time_instances = 100
    # n_parameter_values = 9
    # n_parameters = 3

    # N, M = n_state_variables*n_nodes, n_time_instances*n_parameter_values
    # reduced_rank = M

    # path = "/home/fenics/shared/doc/"
    # points = np.load(path+"points.npy")
    # V = np.load(path+"values.npy")
    # print(points.shape)
    # print(V.shape)
    # points.shape = n_time_instances*n_parameter_values, n_parameters
    # points_n, points_min, points_range = normalise(points, axis=0)
    # l = points_n[1] == 0.4827586206896552
    # print(points)
    # points.shape = n_time_instances, n_parameter_values, n_parameters

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # for i in range(n_parameter_values):
    #     t = points[:, i, 0]
    #     dT = points[:, i, 1]
    #     q = points[:, i, 2]
    #     ax1.plot(t, dT)
    #     ax2.plot(t, q)
    # dT = t/2000*250+50
    # q = total_energy(t, dT)
    # ax1.plot(t, dT, "r--")
    # ax2.plot(t, q, "r--")
    # ax1.set_title("wall temperature difference")
    # ax2.set_title("total heat change")
    # plt.savefig("./tst.png")
    # xi = np.concatenate((t[:, None], dT[:, None], q[:, None]), axis=1)
    # xi_n = (xi-points_min)/points_range

    # V_interpolated = np.zeros((len(xi), reduced_rank))
    # for i in range(reduced_rank):
    #     # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
    #     # vals = V[:, i].numpy().copy()
    #     vals = V[:, i].copy()
    #     V_interpolated[:, i] = griddata(points_n, vals, xi_n, method='linear').copy()
