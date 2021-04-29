
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
        P = np.zeros((d1, d2, D))
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
            k, A = 0.001, 1.0
            dt = np.diff(time, prepend=0)
            P[:, i, 0] = time
            P[:, i, 1] = dT
            P[:, i, 2] = np.cumsum(k*A*dT*dt)
        np.save(path+fileX, X)
        np.save(path+fileP, P)
    return X, P


class POD():
    def __init__(self, X, P):
        (s1, s2, d1, d2) = X.shape
        self.n_state_variables = s1
        self.n_nodes = s2
        self.n_time_instances = d1
        self.n_parameter_values = d2
        self.N, self.M = s1*s2, d1*d2
        # self.n_parameters = 1  # 
        return


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
    X, P = get_snapshot_matrix()
    print(X.shape, P.shape)
