#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:57:34 2021

@author: florianma
source ~/anaconda3/bin/activate root
cd /home/florianma@ad.ife.no/Documents/Repositories/Non-intrusive-reduced-order-modelling-methods/src/
mpiexec -n 4 python SVD_EVP_SLEPc4py.py
"""
import sys
import slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

import timeit
import random
import math


def np_svd(X, full_matrices=False):
    # tic = timeit.default_timer()
    U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    # toc = timeit.default_timer()
    # print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
    #     X.shape[0], X.shape[1], toc - tic))
    return U, S, Vh


# def construct_snapshot_matrix(N, m):
#     """
#     Set N solution of the 1D Laplace problem as columns of a matrix
#     (snapshot matrix).

#     Note: For simplicity we do not perform a linear solve, but use
#     some analytical solution:
#     z(x) = exp(-(x - mu)**2 / sigma)
#     """
#     snapshots = PETSc.Mat().create(PETSc.COMM_SELF)
#     snapshots.setSizes([m, N])
#     snapshots.setType('seqdense')
#     snapshots.setUp()

#     Istart, Iend = snapshots.getOwnershipRange()
#     hx = 1.0 / (m - 1)
#     x_0 = 0.3
#     x_f = 0.7
#     sigma = 0.1**2
#     for i in range(N):
#         mu = x_0 + (x_f - x_0) * random.random()
#         for j in range(Istart, Iend):
#             value = math.exp(-(hx * j - mu)**2 / sigma)
#             snapshots.setValue(j, i, value)
#     snapshots.assemble()

#     return snapshots


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


def project_STS_eigenvectors_to_S_eigenvectors(X_p, Es, m, n, r):
    # Vs = Es.getBV()
    # V_mat = Vs.createMat()
    # U_pc = X_p.matMult(V_mat)
    # print(U_pc.getSizes())
    # print(Vs.getSizes())


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
    # Us.assemble()
    # print(N, sizes)

    # tmpvec2 = X_p.createVecs('left')
    # print(tmpvec2.getSizes())
    print("*********************X_p.getSizes()", X_p.getSizes())
    print("*********************Us.getSizes()", Us.getSizes())
    print("*********************Vs.getSizes()", Vs.getSizes())
    print(tmpvec2.getSizes())
    # print(N)
    for i in range(r):
        # print("view BV ", i, ":")
        tmpvec = Vs.getColumn(i)
        print(tmpvec.getSizes())
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

def sc_svd(X_p):
    XTX = X_p.transposeMatMult(X_p)
    Es = solve_eigenproblem(XTX, r)
    Us, Ss, Vs = project_STS_eigenvectors_to_S_eigenvectors(X_p, Es, m, n, r)
    return Us, Ss, Vs

def sc_svd_direct(X_p, svdtype="trlanczos"):
    Ss = SLEPc.SVD()
    # Ss.setFromOptions()
    Ss.create()
    Ss.setOperator(X_p)
    Ss.setType(svdtype)
    # Ss.setCyclicExplicitMatrix(True)
    # Ss.setImplicitTranspose()
    # -svd_type cross -svd_eps_type lapack -svd_cross_explicitmatrix
    # Ss.setDimensions(X_p.getSize()[0])
    # Ss.setTolerances(tol=.999)
    # nsv: Number of singular values to compute.
    # ncv: Maximum dimension of the subspace to be used by the solver.
    # mpd: Maximum dimension allowed for the projected problem.
    Ss.setDimensions(nsv=r, ncv=2*r)
    Ss.solve()
    Vs, Us = Ss.getBV()  # fast
    return Us, Ss, Vs

t0 = timeit.default_timer()
r = 1800
m, n = 5000, 2000
X_n = np.random.rand(m, n)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
block_size = m // size
s, e = rank*block_size, (rank+1)*block_size
X_p = PETSc.Mat().createDense((m, n), array=X_n[s:e, :].T.reshape(-1))
X_p.assemble()
print("*********************X_p.getSizes()", X_p.getSizes())
t1 = timeit.default_timer()
# X_p = construct_snapshot_matrix(n, m)  # ((200, 200), (30, 30))




t2 = timeit.default_timer()
Us, Ss, Vs = sc_svd_direct(X_p)

t3 = timeit.default_timer()
Un, Sn, Vn = sc2np(Us, Ss, Vs)

t4 = timeit.default_timer()
U, S, VT = np_svd(X_n)

t5 = timeit.default_timer()
print(rank, "of", size)
print("making petsc Mat took \t{:.4}\t seconds.".format(t1 - t0))
print("SVD with slepc took \t{:.4}\t seconds.".format(t3 - t2))
print("slepc to numpy took \t{:.4}\t seconds.".format(t4 - t3))
print("SVD with numpy took \t{:.4}\t seconds.".format(t5 - t4))
# rescale the eigenvectors
# for i in range(r):
#     ll = Es.getEigenvalue(i)
#     print('Eigenvalue ' + str(i) + ': ' + str(ll.real))
#     Us.scaleColumn(i, 1.0 / math.sqrt(ll.real))


# (n_loc, n_glob), (cols, cols_g) = X_p.getSizes()
# r = np.arange(n_loc, dtype=np.int32)
# c = np.arange(cols, dtype=np.int32)
# X_n = X_p.getValues(r, c)

# U, S, Vh = np_svd(X_n)
# # U, S, Vh = np_svd(X_n)
# # U, S, Vh = np_svd(X_n)
# U, S, Vh = np_svd(X_n)
# # U, S, Vh = np_svd(X_n)
# # U, S, Vh = np_svd(X_n)
