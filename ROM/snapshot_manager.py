#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:04:00 2021

@author: florianma
"""
import numpy as np
import os
import matplotlib.pyplot as plt

class Data():
    def __init__(self, X_all, P_all, i_val):
        """
        Splits the data provided into training data and test data.
        The domain as well as the data is normalized.
        The snapshots are expected to be shaped (n_state_variables, n_nodes, n_time_instances, n_parameter_values)
        The points are expected to be shaped (n_time_instances, n_parameter_values, n_parameters)
        The snapshot matrix is reorganised into a shape N*M (n_state_variables*n_nodes, n_time_instances*(n_parameter_values-1))
        The snapshot matrix X is normalized along axis 1 (sized M)
        The domain is normalized along axis 0 (sized M)
        usage:

        my_data = Data(X_all, P_all, 5)
        my_data.test()

        my_ROM = ROM(my_data)
        my_ROM.reduced_rank = 5

        V_rbf = my_ROM.interpolateV(my_data.P_val_n, method="rbf")
        X_pred, X_pred_n = my_ROM.predict(V_rbf)

        error = X_pred-my_data.X_val
        """
        # (s1, s2, d1, d2) = X.shape
        # (d1_, d2_, D) = points.shape

        # self.n_state_variables = s1
        # self.n_nodes = s2
        # self.n_time_instances = d1
        # self.n_parameter_values = d2
        # self.n_parameters = D
        # self.N, self.M = s1*s2, d1*d2
        # self.reduced_rank = self.M

        (self.n_state_variables, self.n_nodes, self.n_time_instances, d2) = X_all.shape
        (d1_, d2_, self.n_parameters) = P_all.shape
        print(self.n_state_variables, self.n_nodes, self.n_time_instances, d2)
        print(d1_, d2_, self.n_parameters)

        assert self.n_time_instances == d1_, "n_time_instances differs."
        assert d2 == d2_, "n_parameter_values differs."
        assert i_val<d2, "test set needs to be 1 out of {:.0f}".format(d2)

        self.n_parameter_values = d2 - 1
        self.N, self.M = self.n_state_variables*self.n_nodes, self.n_time_instances*self.n_parameter_values

        train = np.arange(0, self.n_parameter_values+1) != i_val
        val = ~train

        self.X_train = X_all[... ,train].copy()
        self.P_train = P_all[:, train, :].copy()
        self.X_val = X_all[..., val].copy()
        self.P_val = P_all[:, val, :].copy()
        print(self.X_train.shape)
        print(self.P_train.shape)
        print(self.X_val.shape)
        print(self.P_val.shape)

        N, M = self.n_state_variables*self.n_nodes, self.n_time_instances*self.n_parameter_values
        D, M_ = self.n_parameters, self.n_time_instances*1
        self.X_train.shape = (N, M)
        self.P_train.shape = (M, D)

        self.X_val.shape = (N, M_)
        self.P_val.shape = (M_, D)
        # self.X_train_n.shape = (N, M)
        # self.P_train_n.shape = (M, D)
        # self.X_val_n.shape = (N, M_)
        # self.P_val_n.shape = (M_, D)

        self.X_min, self.X_range = self.bounds(self.X_train, axis=1)
        self.X_train_n = self.normalise(self.X_train, self.X_min, self.X_range)
        self.X_val_n = self.normalise(self.X_val, self.X_min, self.X_range)

        self.P_min, self.P_range = self.bounds(self.P_train, axis=0)
        self.P_train_n = self.normalise(self.P_train, self.P_min, self.P_range)
        self.P_val_n = self.normalise(self.P_val, self.P_min, self.P_range)
        return

    def test(self):
        assert np.allclose(self.P_train, self.rescaleP(self.P_train_n)), "P_train"
        assert np.allclose(self.P_val, self.rescaleP(self.P_val_n)), "P_val"
        assert np.allclose(self.X_train, self.rescaleX(self.X_train_n)), "X_train"
        assert np.allclose(self.X_val, self.rescaleX(self.X_val_n)), "X_val"
        print(self.P_train_n.min(), self.P_train_n.max())
        print(self.X_train_n.min(), self.X_train_n.max())
        assert self.P_train_n.min() == 0.0
        assert self.P_train_n.max() == 1.0
        assert self.X_train_n.min() == 0.0
        assert self.X_train_n.max() == 1.0

        N, M = self.n_state_variables*self.n_nodes, self.n_time_instances*self.n_parameter_values
        D, M_ = self.n_parameters, self.n_time_instances*1

        assert self.X_train.shape == (N, M)
        assert self.P_train.shape == (M, D)
        assert self.X_val.shape == (N, M_)
        assert self.P_val.shape == (M_, D)
        assert self.X_train_n.shape == (N, M)
        assert self.P_train_n.shape == (M, D)
        assert self.X_val_n.shape == (N, M_)
        assert self.P_val_n.shape == (M_, D)
        print("data test passed.")
        return

    def bounds(self, X, axis):
        X_min = X.min(axis=axis, keepdims=True)
        X_max = X.max(axis=axis, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range < 1e-6] = 1e-6
        return X_min, X_range

    def normalise(self, X, X_min, X_range):
        # return X, 0.0, 1.0
        return (X-X_min)/X_range

    def rescaleX(self, X):
        return self.rescale(X, self.X_min, self.X_range)

    def rescaleP(self, P):
        return self.rescale(P, self.P_min, self.P_range)

    def rescale(self, X, X_min, X_range):
        return X*X_range + X_min


def total_energy(time, dT):
    k, A = 0.001, 1.0
    dt = np.diff(time, prepend=0)
    return np.cumsum(k*A*dT*dt)

def get_snapshot_matrix():
    path = "/home/fenics/shared/doc/"
    fileX = "cavity_solidification_dT_X.npy"
    fileP = "cavity_solidification_dT_P.npy"
    try:
        X = np.load(path+fileX)
        P = np.load(path+fileP)
    except:
        print("load snapshot matrix ...")
        s1, s2 = 4, 3015
        d1, d2 = 100, 11
        D = 3
        X = np.zeros((s1, s2, d1, d2))  # 4, 3015, 100, 10
        P = np.zeros((d1, d2, D))  # 100, 10, 3
        dTs = [0, 10, 25, 50, 70, 75, 100, 150, 200, 250, 300]
        for i, dT in enumerate(dTs):
            mypath = path+"cavity_solidification_dt({:.0f})/".format(dT)
            onlyfiles = [f for f in os.listdir(mypath) if f.endswith(".npy")]
            time = np.load(mypath+onlyfiles[0])[::50]  # 100, 6030
            uv = np.load(mypath+onlyfiles[4])[::50]  # 100, 6030
            p = np.load(mypath+onlyfiles[5])[::50]  # 100, 3015
            t = np.load(mypath+onlyfiles[6])[::50]  # 100, 3015
            print(dT)
            print(uv.min(), uv.max())
            if dT == 0:
                dT = time/2000*250+50
            uv.shape = (100, 3015, 2)
            uv.shape = (100, 2, 3015)
            X[0, :, :, i] = uv[:, 1, :].T
            X[1, :, :, i] = uv[:, 0, :].T
            X[2, :, :, i] = p.T
            X[3, :, :, i] = t.T
            P[:, i, 0] = time
            P[:, i, 1] = dT
            P[:, i, 2] = total_energy(time, dT)
        np.save(path+fileX, X)
        np.save(path+fileP, P)
    return X, P


if __name__ == "__main__":
    print("main")
    X_all, P_all = get_snapshot_matrix()
    my_data = Data(X_all, P_all, 0)
    my_data.test()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    t = my_data.P_train[:, 0]
    dT = my_data.P_train[:, 1]
    q = my_data.P_train[:, 2]
    ax1.plot(t, dT, "g.")
    ax2.plot(t, q, "g.")
    t = my_data.P_val[:, 0]
    dT = my_data.P_val[:, 1]
    q = my_data.P_val[:, 2]
    ax1.plot(t, dT, "r.")
    ax2.plot(t, q, "r.")
    ax1.set_ylabel("wall temperature difference")
    ax2.set_ylabel("total heat change")
    plt.savefig("./domain.png")
