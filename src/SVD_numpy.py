#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:44:03 2021

@author: florianma
"""
# source ~/anaconda3/bin/activate root
# mpiexec -n 4 python /home/florianma@ad.ife.no/Documents/Non-intrusive-reduced-order-modelling-methods/src/test_mpi.py
# mpiexec -n 4 python ./test_mpi.py

from mpi4py import MPI
import numpy as np
import timeit

# comm = MPI.COMM_WORLD
# size = comm.Get_size()  # = number of processes
# rank = comm.Get_rank()  # = rank of current process
# print("size", size)
# print("rank", rank)
# print(MPI.Get_processor_name())
# print()


m, n = 12060, 800

r = min(m, n)

X = np.random.rand(m, n)


def np_svd(X, full_matrices=False):
    tic = timeit.default_timer()
    U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    toc = timeit.default_timer()
    print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
        X.shape[0], X.shape[1], toc - tic))
    return U, S, Vh


U, S, VT = np_svd(X)


# def np_svd2(X):
m, n = X.shape
if m > n:
    tic = timeit.default_timer()
    XTX = np.matmul(np.transpose(X), X)
    S2, V2 = np.linalg.eig(XTX)
    S2 = S2.real
    V2 = V2.real
    S2 = S2**.5
    order2 = np.argsort(-S2)
    S2 = S2[order2]
    V2 = V2[:, order2]
    U2 = np.matmul(X, V2 / S2)

    VT2 = np.transpose(V2)

    S2 = S2[:r]
    VT2 = VT2[:r, :]
    U2 = U2[:, :r]
    toc = timeit.default_timer()
    print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
        X.shape[0], X.shape[1], toc - tic))
else:
    tic = timeit.default_timer()
    XXT = np.matmul(X, np.transpose(X))
    S3, U3 = np.linalg.eig(XXT)
    S3 = S3.real
    U3 = U3.real
    S3 = S3**.5

    order3 = np.argsort(-S3)
    S3 = S3[order3]
    U3 = U3[:, order3]
    VT3 = np.matmul(np.transpose(U3) * 1 / S3[:, None], X)

    S3 = S3[:r]
    VT3 = VT3[:r, :]
    U3 = U3[:, :r]
    toc = timeit.default_timer()
    print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
        X.shape[0], X.shape[1], toc - tic))
