#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:04:00 2021

@author: florianma
"""
import numpy as np
import os
import matplotlib.pyplot as plt


class Data:

    # data handling class. takes care of
    #     - normalising / scaling
    #     - splitting into train and test data
    #     -

    def __init__(self, X, xi, x, y, tri, dims, phase_length=None):
        self.X = X  # snapshots
        self.xi = xi  # parameterspace
        self.x = x  # mesh vertices (x)
        self.y = y  # mesh vertices (y)
        self.tri = tri  # mesh triangles
        # n: number of nodes (s1) * number of physical quantities (s2)
        # m: number of snapshots = num datasets (d2) * snapshots_per_dataset (d1)
        # r: truncation rank
        # D: dimension parameterspace (2)
        # U_x snapshot matrix. SVD: X = U*S*V
        # X: (n, m) = (s1*s2, d1*d2)
        # xi: (m, D) = (d1*d2, D)
        [[s1, s2], [d1, d2]] = dims
        self.s1, self.s2, self.d1, self.d2 = s1, s2, d1, d2
        self.dimensions = dims
        self.phase_length = phase_length
        self.normalise()

    def test_my_data(self):
        n, m = self.s1 * self.s2, self.d1*self.d2
        assert self.X.shape == (n, m)
        assert self.X_n.shape == (n, m)
        assert self.xi.shape == (m, 2)
        # check if the grid is alright
        assert self.mgrid[0].size == self.xi[..., 0].size
        assert self.mgrid[1].size == self.xi[..., 1].size
        assert np.allclose(self.mgrid[0].ravel(), self.xi[..., 0])
        assert np.allclose(self.mgrid[1].ravel(), self.xi[..., 1])
        # bounds
        # assert self.X_n.min() == 0.0  # not true for test data!
        # assert self.X_n.max() == 1.0  # not true for test data!
        return

    def test_scaling(self):
        # this might take a while...
        assert np.allclose(self.scale_up(self.X_n), self. X)
        return

    def get_grid(self, dim=2):
        p1, p2 = np.unique(self.xi[..., 0]), np.unique(self.xi[..., 1])
        self.mgrid = np.meshgrid(p1, p2, indexing="ij")
        self.grid = [p1, p2]
        self.test_my_data()
        return

    def normalise(self):
        X_min = self.X.min(axis=1)[:, None]  # n
        X_max = self.X.max(axis=1)[:, None]  # n
        X_range = X_max - X_min
        X_range[X_range < 1e-6] = 1e-6
        self.X_min = X_min
        self.X_range = X_range
        self.X_n = self.scale_down(self.X)
        return self.X_n

    def scale_like(self, some_data):
        self.X_min = some_data.X_min
        self.X_range = some_data.X_range
        self.X_n = self.scale_down(self.X)
        return self.X_n

    def scale_down(self, Snapshots):
        return (Snapshots-self.X_min)/self.X_range

    def scale_up(self, Snapshots_n):
        return Snapshots_n * self.X_range + self.X_min


def split(data, set_i, phase_length=None):
    # TODO: find out why its so slow
    # [[s1, s2], [d1, d2]] = self.dims
    data.X.shape = (data.s1, data.s2, data.d1, data.d2)
    data.xi.shape = (data.d1, data.d2, 2)

    i_train = np.delete(np.arange(data.d2), set_i)
    data.X_train = data.X[..., i_train].copy()
    data.X_valid = data.X[..., set_i].copy()
    data.xi_train = data.xi[:, i_train, :].copy()
    data.xi_valid = data.xi[:, set_i, :].copy()

    n_p = 1
    if isinstance(data.phase_length, np.ndarray):
        offset = np.c_[phase_length[i_train],
                       np.zeros_like(phase_length[i_train])]
        data.xi_train = np.concatenate((data.xi_train-offset,
                                        data.xi_train,
                                        data.xi_train+offset), axis=0)
        # X_train = np.concatenate((X_train, X_train, X_train), axis=2)
        n_p = 3  # number of repetitions of each periods
    data.X.shape = (data.s1*data.s2, data.d1*data.d2)
    data.xi.shape = (data.d1*data.d2, 2)
    data.X_train.shape = (data.s1*data.s2, data.d1*(data.d2-1))
    data.xi_train.shape = (n_p*data.d1*(data.d2-1), 2)
    data.X_valid.shape = (data.s1*data.s2, data.d1*1)
    data.xi_valid.shape = (data.d1*1, 2)

    train_data = Data(X, xi, x, y, tri, dims, phase_length=None)
    return data.X_train, data.X_valid, data.xi_train, data.xi_valid


class Data_old():
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

        (self.n_state_variables, self.n_nodes,
         self.n_time_instances, d2) = X_all.shape
        (d1_, d2_, self.n_parameters) = P_all.shape
        print(self.n_state_variables, self.n_nodes, self.n_time_instances, d2)
        print(d1_, d2_, self.n_parameters)

        assert self.n_time_instances == d1_, "n_time_instances differs."
        assert d2 == d2_, "n_parameter_values differs."
        assert i_val < d2, "test set needs to be 1 out of {:.0f}".format(d2)

        self.n_parameter_values = d2 - 1
        self.N, self.M = self.n_state_variables * \
            self.n_nodes, self.n_time_instances*self.n_parameter_values

        train = np.arange(0, self.n_parameter_values+1) != i_val
        val = ~train

        self.X_train = X_all[..., train].copy()
        self.P_train = P_all[:, train, :].copy()
        self.X_val = X_all[..., val].copy()
        self.P_val = P_all[:, val, :].copy()
        print(self.X_train.shape)
        print(self.P_train.shape)
        print(self.X_val.shape)
        print(self.P_val.shape)

        N, M = self.n_state_variables * \
            self.n_nodes, self.n_time_instances*self.n_parameter_values
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
        assert np.allclose(self.P_train, self.rescaleP(
            self.P_train_n)), "P_train"
        assert np.allclose(self.P_val, self.rescaleP(self.P_val_n)), "P_val"
        assert np.allclose(self.X_train, self.rescaleX(
            self.X_train_n)), "X_train"
        assert np.allclose(self.X_val, self.rescaleX(self.X_val_n)), "X_val"
        print(self.P_train_n.min(), self.P_train_n.max())
        print(self.X_train_n.min(), self.X_train_n.max())
        assert self.P_train_n.min() == 0.0
        assert self.P_train_n.max() == 1.0
        assert self.X_train_n.min() == 0.0
        assert self.X_train_n.max() == 1.0

        N, M = self.n_state_variables * \
            self.n_nodes, self.n_time_instances*self.n_parameter_values
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


def load_snapshots_cylinder(path):
    x = np.load(path+"x.npy")
    y = np.load(path+"y.npy")
    tri = np.load(path+"tri.npy")
    time = np.load(path+"Re020_time.npy")
    Res = np.array([20, 33, 50, 60, 75, 100, 125, 150, 200])
    s1 = 4  # u, v, p and T
    s2 = len(x)
    d1 = len(time)
    d2 = len(Res)
    n, m = s1*s2, d1*d2
    dimensions = [[s1, s2], [d1, d2]]
    D = len(dimensions[1])  # time and wall temperature
    print("n physical quantities", s1)
    print("n_nodes", s2)
    print("snapshots_per_dataset", d1)
    print("n_datasets", d2)
    U = np.zeros((s1, s2, d1, d2))
    xi = np.zeros((d1, d2, D))
    for i, Re in enumerate(Res):  # iteration along d2
        u = np.load(path+"Re{:03.0f}_velocity_u.npy".format(Re))
        v = np.load(path+"Re{:03.0f}_velocity_v.npy".format(Re))
        time = np.load(path+"Re{:03.0f}_time.npy".format(Re))
        # t_min, t_max = time.min(), time.max()
        # t_range = t_max-t_min
        # time = (time-t_min)/t_range
        p = np.load(path+"Re{:03.0f}_pressure.npy".format(Re))
        U[0, :, :, i] = u
        U[1, :, :, i] = v
        U[2, :, :, i] = p
        # U[3, :, :, i] = temp
        xi[:, i, 0] = time
        xi[:, i, 1] = Re
        print(Re, ":", p.shape, len(time))
    U.shape = (n, m)
    xi.shape = (m, D)
    te = xi[:, 0].reshape(-1, 9).T[:, -1]
    ts = xi[:, 0].reshape(-1, 9).T[:, 0]
    phase_length = te-ts
    return U, xi, x, y, tri, dimensions, phase_length


def load_snapshots_cavity(path):
    x = np.load(path+"x.npy")
    y = np.load(path+"y.npy")
    tri = np.load(path+"tri.npy")
    time = np.load(path+"Tamb400_time.npy")
    Ts = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625])
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
    X = np.zeros((s1, s2, d1, d2))
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
        X[0, :, :, i] = u
        X[1, :, :, i] = v
        X[2, :, :, i] = p
        X[3, :, :, i] = temp
        xi[:, i, 0] = time
        xi[:, i, 1] = t_amb
        print(t_amb, ":", p.shape, len(time))
    X.shape = (n, m)
    xi.shape = (m, D)
    return X, xi, x, y, tri, dimensions, None


if __name__ == "__main__":
    path = "C:/Users/florianma/Documents/data/freezing_cavity/"
    # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
    my_data = Data(*load_snapshots_cavity(path))

    asd
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
