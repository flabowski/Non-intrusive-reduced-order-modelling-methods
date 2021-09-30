#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:57:34 2021

@author: florianma
"""
import sys
import slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

import random
import math


def np_svd(X, full_matrices=False):
    # tic = timeit.default_timer()
    U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    # toc = timeit.default_timer()
    # print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
    #     X.shape[0], X.shape[1], toc - tic))
    return U, S, Vh


def construct_snapshot_matrix(N, m):
    """
    Set N solution of the 1D Laplace problem as columns of a matrix
    (snapshot matrix).

    Note: For simplicity we do not perform a linear solve, but use
    some analytical solution:
    z(x) = exp(-(x - mu)**2 / sigma)
    """
    snapshots = PETSc.Mat().create(PETSc.COMM_SELF)
    snapshots.setSizes([m, N])
    snapshots.setType('seqdense')
    snapshots.setUp()

    Istart, Iend = snapshots.getOwnershipRange()
    hx = 1.0 / (m - 1)
    x_0 = 0.3
    x_f = 0.7
    sigma = 0.1**2
    for i in range(N):
        mu = x_0 + (x_f - x_0) * random.random()
        for j in range(Istart, Iend):
            value = math.exp(-(hx * j - mu)**2 / sigma)
            snapshots.setValue(j, i, value)
    snapshots.assemble()

    return snapshots


def solve_eigenproblem(snapshots, N):
    """
    Solve the eigenvalue problem: the eigenvectors of this problem form the
    POD basis.
    """
    print('Solving POD basis eigenproblem using eigensolver...')

    Es = SLEPc.EPS()
    Es.create(PETSc.COMM_SELF)
    Es.setDimensions(N)
    Es.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    Es.setTolerances(1.0e-8, 500)
    Es.setKrylovSchurRestart(0.6)
    Es.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    Es.setOperators(snapshots)
    Es.setFromOptions()

    Es.solve()
    print('Solved POD basis eigenproblem.')
    return Es


def project_STS_eigenvectors_to_S_eigenvectors(Es, m, n, r):
    Vs = Es.getBV()
    Vs.setActiveColumns(0, r)
    # m, n, r = 200, 30, 8
    Ss = PETSc.Vec().createSeq(m)
    tmpvec2 = PETSc.Vec().createSeq(m)

    # sizes = X_p.getSizes()[0]
    # N = Vs.getActiveColumns()[1]
    Us = SLEPc.BV().create(PETSc.COMM_SELF)
    Us.setSizes(m, r)
    Us.setActiveColumns(0, r)
    Us.setFromOptions()
    # print(N, sizes)

    # tmpvec2 = X_p.createVecs('left')
    # print(tmpvec2.getSizes())
    # print(bvEs.getSizes())
    # print(N)
    for i in range(r):
        # print("view BV ", i, ":")
        tmpvec = Vs.getColumn(i)
        # tmpvec.view()
        X_p.mult(tmpvec, tmpvec2)
        Us.insertVec(i, tmpvec2)
        Vs.restoreColumn(i, tmpvec)

        ll = Es.getEigenvalue(i)
        # print('Eigenvalue ' + str(i) + ': ' + str(ll.real))
        Ss[i] = math.sqrt(ll.real)
        Us.scaleColumn(i, 1.0 / Ss[i])
    print(Us.getSizes())
    print(Vs.getSizes())
    return Us, Ss, Vs

# Vs, Us = Ss.getBV()  # fast


def sc2np(Us, Ss, Vs):

    (nU, n_glob), cols = Us.getSizes()
    (nV, n_glob), cols = Vs.getSizes()
    # N = bv.getActiveColumns()[1]
    Un = np.empty((nU, r),)
    Sn = np.empty(r,)
    Vn = np.empty((nV, r),)
    print(nU, nV)

    # tmpvec = X_p.createVecs('left')
    # print(bvEs.getSizes())
    for i in range(r):
        # print("view BV ", i, ":")
        tmpvec = Us.getColumn(i)
        Un[:, i] = tmpvec.getArray()
        Us.restoreColumn(i, tmpvec)

        tmpvec = Vs.getColumn(i)
        Vn[:, i] = tmpvec.getArray()
        Vs.restoreColumn(i, tmpvec)

        Sn[i] = Ss.getValue(i)
    return Un, Sn, Vn


r = 30
m, n = 200, 30
X_n = np.random.rand(m, n)
X_p = PETSc.Mat().createDense((m, n), array=X_n.T.reshape(-1))
# X_p = construct_snapshot_matrix(n, m)  # ((200, 200), (30, 30))


def sc_svd(X_p):
    XTX = X_p.transposeMatMult(X_p)
    Es = solve_eigenproblem(XTX, r)
    Us, Ss, Vs = project_STS_eigenvectors_to_S_eigenvectors(Es, m, n, r)
    return Us, Ss, Vs


Us, Ss, Vs = sc_svd(X_p)
Un, Sn, Vn = sc2np(Us, Ss, Vs)

U, S, VT = np_svd(X_n)

# rescale the eigenvectors
# for i in range(r):
#     ll = Es.getEigenvalue(i)
#     print('Eigenvalue ' + str(i) + ': ' + str(ll.real))
#     Us.scaleColumn(i, 1.0 / math.sqrt(ll.real))


(n_loc, n_glob), (cols, cols_g) = X_p.getSizes()
r = np.arange(n_loc, dtype=np.int32)
c = np.arange(cols, dtype=np.int32)
X_n = X_p.getValues(r, c)

U, S, Vh = np_svd(X_n)
# U, S, Vh = np_svd(X_n)
# U, S, Vh = np_svd(X_n)
U, S, Vh = np_svd(X_n)
# U, S, Vh = np_svd(X_n)
# U, S, Vh = np_svd(X_n)
