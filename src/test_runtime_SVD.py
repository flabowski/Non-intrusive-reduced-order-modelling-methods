# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:17:01 2021

@author: florianma
"""
from nirom.low_rank_model_construction.basis_function_interpolation import BasisFunctionRegularGridInterpolator as BFRGI
from nirom.low_rank_model_construction.proper_orthogonal_decomposition import truncate_basis, row_svd
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

if __name__ == "__main__":
    # timing the SVD of a nxm matrix
    # 1. set m = 5000 and vary n (100 ... 12 000)
    # 2. set n = 5000 and vary m (100 ... 12 000)
    # 3. load real dataset (n = 12060) and vary m by randomly picking snapshots
    # m = 100 ... 1000, m = 1000 ... 25 000, m = 25 000 ... 50 000
    # 4. err vs. rank and err vs. eps with and without randomply choosing ss
    N = 20
    repetitions = 3
    # N = 11
    # repetitions = 3

    eps = .9999
    m = 12060
    # y = np.zeros(N,)
    # X = np.random.rand(m, 12000)
    rand_frames = np.random.randint(0, 50000, 40000)
    # 12060 * 50000
    X = np.load(
        "C:/Users/florianma/Documents/data/interpolation1/X.npy")[:, rand_frames]
    y = np.zeros((N, repetitions))

    # for i, n in enumerate(np.linspace(100, 2000, N, dtype=np.int32)):
    # for i, n in enumerate(np.linspace(10, 1000, 100, dtype=np.int32)):
    # for i, n in enumerate(np.linspace(1000, 25000, 25, dtype=np.int32)):
    for i, n in enumerate(np.linspace(25000, 50000, 6, dtype=np.int32)):
        print(n, end="\t")
        for j in range(repetitions):
            Xi = X[:m, :n]
            t0 = timeit.default_timer()
            if n <= 5000:
                U_, S_, VT_ = truncate_basis(*svd(Xi), eps)
                t1 = timeit.default_timer()
                X_approx = matmul(U_*S_, VT_)
                error = np.std(X_approx-Xi)
                # X_approx2 = matmul(U_, matmul(transpose(U_), Xi))
            else:
                error = -1
                t1 = timeit.default_timer()
            print(t1-t0, "\t", error, end="\t")
            for col in [2, 4, 8]:
                t1 = timeit.default_timer()
                if (n / col <= 5000) and (n/col > 50):
                    U_, S_, VT_ = row_svd(
                        Xi, col, eps=eps, ommit_V=True, QR_DECOMPOSITION=True)

                    t2 = timeit.default_timer()
                    X_approx = matmul(U_*S_, VT_)
                    error = np.std(X_approx-Xi)
                    # X_approx2 = matmul(U_, matmul(transpose(U_), Xi))
                else:
                    error = -1
                    t2 = timeit.default_timer()
                t2 = timeit.default_timer()
                print(t2-t1, "\t", error, end="\t")

                # U, S, VT = row_svd(Xi, 1, eps=eps, ommit_V=False, QR_DECOMPOSITION=True)

                # t3 = timeit.default_timer()
                # U, S, VT = row_svd(Xi, 1, eps=eps, ommit_V=False, QR_DECOMPOSITION=False)

                # t4 = timeit.default_timer()
                # U, S, VT = row_svd(Xi, 1, eps=eps, ommit_V=True, QR_DECOMPOSITION=False)
                # t5 = timeit.default_timer()
                # print(t2-t1, t3-t2, t4-t3, t5-t4, sep="\t", end="\t")

            # y[i, j] = t1-t0
            # print(y[i, j], end="\t")
        # print(np.mean(y[i, :]))
        print()
