
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import isfile, join
import pickle
import tensorflow as tf
from scipy.interpolate import Rbf
import timeit
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
# import seaborn as sns
# import pandas as pd
# from mayavi import mlab
np.set_printoptions(suppress=True)
plot_width = 16


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
        d1, d2 = 100, 10
        D = 3
        X = np.zeros((s1, s2, d1, d2))  # 4, 3015, 100, 10
        P = np.zeros((d1, d2, D))  # 100, 10, 3
        dTs = [10, 25, 50, 70, 75, 100, 150, 200, 250, 300]
        for i, dT in enumerate(dTs):
            mypath = path+"cavity_solidification_dt({:.0f})/".format(dT)
            onlyfiles = [f for f in os.listdir(mypath) if f.endswith(".npy")]
            time = np.load(mypath+onlyfiles[0])[::50]  # 100, 6030
            uv = np.load(mypath+onlyfiles[4])[::50]  # 100, 6030
            p = np.load(mypath+onlyfiles[5])[::50]  # 100, 3015
            t = np.load(mypath+onlyfiles[6])[::50]  # 100, 3015
            uv.shape = (100, 3015, 2)
            X[0, :, :, i] = uv[:, :, 0].T
            X[1, :, :, i] = uv[:, :, 1].T
            X[2, :, :, i] = p.T
            X[3, :, :, i] = t.T
            P[:, i, 0] = time
            P[:, i, 1] = dT
            P[:, i, 2] = total_energy(time, dT)
        np.save(path+fileX, X)
        np.save(path+fileP, P)
    return X, P




def normalise(X, axis=0):
    # return X, 0.0, 1.0
    X_min = X.min(axis=axis, keepdims=True)
    X_max = X.max(axis=axis, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-6] = 1e-6
    X_n = (X-X_min)/X_range
    return X_n, X_min, X_range


class ROM():
    def __init__(self, X, points):
        (s1, s2, d1, d2) = X.shape
        (d1_, d2_, D) = points.shape
        print(s1, s2, d1, d2)
        print(d1_, d2_, D)

        self.n_state_variables = s1
        self.n_nodes = s2
        self.n_time_instances = d1
        self.n_parameter_values = d2
        self.n_parameters = D
        self.N, self.M = s1*s2, d1*d2
        self.reduced_rank = self.M
        print(X.shape, P.shape)
        print(self.N, self.M)
        print(X.shape)

        X.shape = (self.N, self.M)
        points.shape = (self.M, D)
        self.X, self.points = X, points
        self.X_n, self.X_min, self.X_range = normalise(X, axis=1)
        self.points_n, self.points_min, self.points_range = normalise(points, axis=0)
        S, U, V = tf.linalg.svd(self.X_n, full_matrices=False)  # (N, R), (R, R), (M, M)

        path = "/home/fenics/shared/doc/"
        np.save(path+"points.npy", self.points_n.reshape(d1, d2, D))
        np.save(path+"values.npy", V)
        self.S = S.numpy()
        self.U = U.numpy()
        self.V = V.numpy()
        return

    def interpolateV(self, xi):
        """
        Parameters
        ----------
        xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
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

        V_interpolated = np.zeros((self.n_time_instances, self.reduced_rank))
        for i in range(self.reduced_rank):
            # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
            vals = self.V[:, i].copy()
            V_interpolated[:, i] = griddata(self.points_n, vals, xi, method='linear').copy()
        self.V_interpolated = V_interpolated
        return V_interpolated

    def predict(self, V_):
        r = self.reduced_rank
        U_hat = self.U[:, :r]  # (n, r)
        S_hat = self.S[:r]  # (r, r) / (r,) wo  non zero elements
        V_hat = V_[:, :r]  # (d1, r)
        X_approx_n = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
        X_approx = X_approx_n*self.X_range + self.X_min
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

    # test 1
    train = np.array([0,1,2,3,4,6,7,8,9])
    val = np.array([5])
    X = X_all[... ,train].copy()
    P = P_all[:, train, :].copy()
    X_val = X_all[..., val].copy()
    P_val = P_all[:, val, :].copy()

    print(X.shape, P.shape)
    my_ROM = ROM(X, P)
    P_val_n = normalise_P  # P_val-points_min)/points_range
    V_i = my_ROM.interpolateV(P_val_n)
    X_pred, X_pred_n = my_ROM.predict(V_i)
    print(X_pred)
    print(X_val)
    asd
    print("S")

    n_state_variables = 4
    n_nodes = 3015
    n_time_instances = 100
    n_parameter_values = 9
    n_parameters = 3

    N, M = n_state_variables*n_nodes, n_time_instances*n_parameter_values
    reduced_rank = M

    path = "/home/fenics/shared/doc/"
    points = np.load(path+"points.npy")
    V = np.load(path+"values.npy")
    print(points.shape)
    print(V.shape)
    points.shape = n_time_instances*n_parameter_values, n_parameters
    points_n, points_min, points_range = normalise(points, axis=0)
    l = points_n[1] == 0.4827586206896552
    print(points)
    points.shape = n_time_instances, n_parameter_values, n_parameters
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(n_parameter_values):
        t = points[:, i, 0]
        dT = points[:, i, 1]
        q = points[:, i, 2]
        ax1.plot(t, dT)
        ax2.plot(t, q)
    dT = t/2000*250+50
    q = total_energy(t, dT)
    ax1.plot(t, dT, "r--")
    ax2.plot(t, q, "r--")
    ax1.set_title("wall temperature difference")
    ax2.set_title("total heat change")
    plt.savefig("./tst.png")
    xi = np.concatenate((t[:, None], dT[:, None], q[:, None]), axis=1)
    xi_n = (xi-points_min)/points_range

    V_interpolated = np.zeros((len(xi), reduced_rank))
    for i in range(reduced_rank):
        # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
        # vals = V[:, i].numpy().copy()
        vals = V[:, i].copy()
        V_interpolated[:, i] = griddata(points_n, vals, xi_n, method='linear').copy()
