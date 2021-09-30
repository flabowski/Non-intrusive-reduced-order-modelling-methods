#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:06:28 2021

@author: florianma
source ~/anaconda3/bin/activate root
mpiexec -n 4 python /home/florianma@ad.ife.no/Documents/Non-intrusive-reduced-order-modelling-methods/src/test_mpi.py
mpiexec -n 4 python ./demo_slepc.py
python demo_slepc.py -eps_nev 10 -eps_tol 1e-11
mpiexec -n 1 python ./demo_slepc.py -n 40 -eps_nev 16 -eps_view

"""
# The first thing to do is initialize the libraries. This is normally not
# required, as it is done automatically at import time. However, if you want
# to gain access to the facilities for accesing command-line options, the
# following lines must be executed by the main script prior to any petsc4py
# or slepc4py calls:
import sys
import slepc4py
slepc4py.init(sys.argv)
import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc
import timeit

opts = PETSc.Options()
X_np = np.load("/home/florianma@ad.ife.no/Documents/cavity/X4.npy")

m = opts.getInt('m', 12060)
n = opts.getInt('n', 300)
r = opts.getInt('r', 300)
print("m x n:", m, "x", n)
print("number eigen values:", r)

# Instead of solving the SVD of S, we solve the standard
# eigenvalue problem on X.T*X
# this assumes that m > n (tall- skinny matrices)

# numpy implementation:

tic = timeit.default_timer()
X = X_np[:m, :n]
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

# print("nev", nev)


X_pc = PETSc.Mat().createDense((m, n), array=X_np[:m, :n].T.reshape(-1))
X_pc.setFromOptions()
X_pc.setUp()
rstart, rend = X_pc.getOwnershipRange()
print("getOwnershipRange:", rstart, rend)
X_pc.assemble()

XtX = X_pc.transposeMatMult(X_pc)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# eigenvalue problem solver
E = SLEPc.EPS()
E.create()
E.setOperators(XtX)
E.setDimensions(r, PETSc.DECIDE)     # set number of  eigenvalues to compute
E.setProblemType(SLEPc.EPS.ProblemType.HEP)  # SLEPc.EPS.ProblemType.NHEP
# E.setTolerances(1.0e-8, 500)
# E.setKrylovSchurRestart(0.6)
# E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
E.setFromOptions()
E.solve()

nconv = E.getConverged()
print('Number of converged eigenvalues: %i' % nconv)
# E.view()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   X    =   U    S   VT
# (m, n) = (m, r) r, (r, n)
V_pc = E.getBV()
V_pc.setActiveColumns(0, r)


sizes = X_pc.getSizes()[0]
N = V_pc.getActiveColumns()[1]
U_pc = SLEPc.BV().create()  # PETSc.COMM_SELF
U_pc.setSizes(sizes, N)
U_pc.setActiveColumns(0, N)
U_pc.setFromOptions()

print("N == r?", N, r)
print("sizes == m, r?", sizes, m, r)
# X_pc.getSize(): (12060, 300)
# U_pc.getSizes(): (12060, 12060)
# V_pc.getSizes(): (300, 300)
# tmpvec.getSize():    300
# tmpvec2.getSize():  12060
# tmpvec3.getSize()   300
tmpvec3, tmpvec2 = X_pc.getVecs()
for i in range(N):
    print(i)
    tmpvec = V_pc.getColumn(i)  # 300
    X_pc.mult(tmpvec, tmpvec2)
    U_pc.insertVec(i, tmpvec2)
# #     print("attempting to restore")
#     V_pc.restoreColumn(i, tmpvec)

# alternative?
V_mat = V_pc.createMat()
U_pc = X_pc.matMult(V_mat)
U_pc.getSizes()
V_pc.getSizes()

# sizes = X_pc.getSizes()[0]
# N = V_pc.getActiveColumns()[1]

# U = SLEPc.BV().create()
# U.setSizes(sizes, N)
# U.setActiveColumns(0, N)
# U.setFromOptions()


# print(V_pc.getSizes())

# tmpvec2 = X_pc.createVecs('left')
# for i in range(N):
#     tmpvec = V_pc.getColumn(i)
#     X_pc.mult(tmpvec, tmpvec2)
#     U.insertVec(i, tmpvec2)
#     V_pc.restoreColumn(i, tmpvec)


# Print = PETSc.Sys.Print
#
# Print()
# Print("******************************")
# Print("*** SLEPc Solution Results ***")
# Print("******************************")
# Print()

# its = E.getIterationNumber()
# Print("Number of iterations of the method: %d" % its)

# eps_type = E.getType()
# Print("Solution method: %s" % eps_type)

# nev, ncv, mpd = E.getDimensions()
# Print("Number of requested eigenvalues: %d" % nev)

# tol, maxit = E.getTolerances()
# Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

# nconv = E.getConverged()
# Print("Number of converged eigenpairs %d" % nconv)

# if nconv > 0:
#     # Create the results vectors
#     vr, wr = X_pc.getVecs()
#     vi, wi = X_pc.getVecs()
#     #
#     Print()
#     Print("        k          ||Ax-kx||/||kx|| ")
#     Print("----------------- ------------------")
#     for i in range(nconv):
#         k = E.getEigenpair(i, vr, vi)
#         error = E.computeError(i)
#         if k.imag != 0.0:
#             Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
#         else:
#             Print(" %12f      %12g" % (k.real, error))
# Print()
