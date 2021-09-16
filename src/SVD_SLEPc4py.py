#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:30:44 2021

@author: florianma
    https://slepc.upv.es/documentation/slepc.pdf
    https://lists.mcs.anl.gov/pipermail/petsc-users/2012-May/013379.html
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.732.5862&rep=rep1&type=pdf
    https://slepc.upv.es/handson/handson4.html
    https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg40589.html
"""
# source ~/anaconda3/bin/activate root
# mpiexec -n 4 /home/florianma@ad.ife.no/Documents/Non-intrusive-reduced-order-modelling-methods/src/test_slepc4py_SVD.py

# import matplotlib
# import sys, slepc4py
# slepc4py.init(sys.argv)
# from petsc4py import PETSc
# from slepc4py import SLEPc
# import numpy as np
import timeit
import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc
import numpy as np
# import pylab
import matplotlib.pyplot as plt
import math
from ROM.snapshot_manager import Data, load_snapshots_cavity
plt.close("all")
if "X" not in locals():
    X = np.load("/home/florianma@ad.ife.no/Documents/cavity/X4.npy")
    print("load X")
    # for s in range(10)
    # np.save("/home/florianma@ad.ife.no/Documents/cavity/X{:.0f}.npy".format(s), X[:, (s-1)*5000:s*5000])


def np_svd(X, full_matrices=False):
    U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    return U, S, Vh

# opts = PETSc.Options()
# n = opts.getInt('n', 30)

# A = PETSc.Mat().create()
# A.setSizes([n, n])
# A.setFromOptions()
# A.setUp()

# rstart, rend = A.getOwnershipRange()

# # first row
# if rstart == 0:
#     A[0, :2] = [2, -1]
#     rstart += 1
# # last row
# if rend == n:
#     A[n-1, -2:] = [-1, 2]
#     rend -= 1
# # other rows
# for i in range(rstart, rend):
#     A[i, i-1:i+2] = [-1, 2, -1]

# A.assemble()
# U = PETSc.Vec()
# [0]PETSC ERROR: ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# [0]PETSC ERROR: Caught signal number 11 SEGV: Segmentation Violation, probably memory access out of range
# [0]PETSC ERROR: Try option ‑start_in_debugger or ‑on_error_attach_debugger
# [0]PETSC ERROR: or see https://www.mcs.anl.gov/petsc/documentation/faq.html#valgrind
# [0]PETSC ERROR: or try http://valgrind.org on GNU/linux and Apple Mac OS X to find memory corruption errors
# [0]PETSC ERROR: configure using ‑‑with‑debugging=yes, recompile, link, and run
# [0]PETSC ERROR: to get more information on the crash.
# application called MPI_Abort(MPI_COMM_WORLD, 59) ‑ process 0
# [unset]: write_line error; fd=𔂫 buf=:cmd=abort exitcode=59
# :
# system msg for write_line failure : Bad file descriptor


def np2pc(X_n):
    m, n = X_n.shape
    # r = np.arange(shape[0], dtype=np.int32)
    # c = np.arange(shape[1], dtype=np.int32)

    # X_p1= PETSc.Mat().create()
    # X_p1.setSizes(shape)
    # X_p1.setUp()
    # for row in range(X_n.shape[0]):
    #     for col in range(X_n.shape[1]):
    #         X_p1.setValue(row, col, X_n[row, col])
    # X_p1.assemble()

    # t2 = timeit.default_timer()

    # r = np.arange(shape[0], dtype=np.int32)
    # c = np.arange(shape[1], dtype=np.int32)
    # X_p2 = PETSc.Mat().create()
    # X_p2.setSizes(shape)
    # X_p2.setUp()
    # X_p2.setValues(r, c, X_n)
    # X_p2.assemble()

    X_p = PETSc.Mat().createDense((m, n), array=X_n.T.reshape(-1))
    # opts = PETSc.Options()
    # X_p = PETSc.Mat().createDense((m, n))
    X_p.setFromOptions()
    X_p.setUp()
    # rstart, rend = X_p.getOwnershipRange()
    X_p.assemble()
    # vec = X_p.getDenseArray()
    return X_p


def BV2np(Us, Ss, Vs, nsv):
    # option 1:
    # r = np.arange(n_loc, dtype=np.int32)
    # c = np.arange(nsv, dtype=np.int32)
    # Un = Us.createMat().getValues(r, c)
    # option 2:
    # mat, _ = Us.getMatrix()
    # mat.createAIJ(size=(n_loc,cols), csr=(ai,aj,aa))
    # mat.getValuesCSR()
    # option 3
    # Un = np.empty((nU, nsv),)
    # Us_mat = Us.createMat()
    # for col in range(k):
    #     # Vs.getColumn crashes sometimes
    #     Un[:, col] = Us_mat.getColumnVector(col).array
    # option 4:
    # for i in range(nsv):
    #     tmpvec = Us.getColumn(i)
    #     Un[:, i] = tmpvec.getArray()
    #     Us.restoreColumn(i, tmpvec)

    (nU, nU_glob), colsU = Us.getSizes()
    (nV, nV_glob), colsV = Vs.getSizes()
    Un = np.empty((nU, nsv),)
    Sn = np.empty(nsv,)
    Vn = np.empty((nV, nsv),)
    for i in range(nsv):
        tmpvec = Us.getColumn(i)
        Un[:, i] = tmpvec.getArray()
        Us.restoreColumn(i, tmpvec)
        tmpvec = Vs.getColumn(i)
        Vn[:, i] = tmpvec.getArray()
        Vs.restoreColumn(i, tmpvec)
        Sn[i] = Ss.getValue(i)
    return Un, Sn, Vn


def solve_eigenproblem(X_p, N):
    """
    Solve the eigenvalue problem: the eigenvectors of this problem form the
    POD basis.
    """
    # print('Solving POD basis eigenproblem using eigensolver...')
    Es = SLEPc.EPS()
    Es.create(PETSc.COMM_SELF)
    Es.setDimensions(N)
    Es.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    # Es.setTolerances(1.0e-8, 500)
    # Es.setKrylovSchurRestart(0.6)
    Es.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    Es.setOperators(X_p)
    Es.setFromOptions()
    Es.solve()
    # print('Solved POD basis eigenproblem.')
    return Es


def getU(Es, X_p, r):
    m, n = X_p.getSize()
    Vs = Es.getBV()
    Vs.setActiveColumns(0, r)
    Ss = PETSc.Vec().createSeq(m)
    tmpvec2 = PETSc.Vec().createSeq(m)

    Us = SLEPc.BV().create(PETSc.COMM_SELF)
    Us.setSizes(m, r)
    Us.setActiveColumns(0, r)
    Us.setFromOptions()
    for i in range(r):
        tmpvec = Vs.getColumn(i)
        X_p.mult(tmpvec, tmpvec2)
        Us.insertVec(i, tmpvec2)
        Vs.restoreColumn(i, tmpvec)
        ll = Es.getEigenvalue(i)
        s = ll.real
        if s > 0:
            Ss[i] = math.sqrt(s)
            Us.scaleColumn(i, 1.0 / Ss[i])
        else:
            Ss[i] = 0
            Us.scaleColumn(i, 0.0)
    return Us, Ss, Vs


def sc_svd2(X_p, nsv):
    m, n = X_p.getSize()
    XTX = X_p.transposeMatMult(X_p)
    Es = solve_eigenproblem(XTX, nsv)
    nconv = Es.getConverged()
    print('Number of converged eigenvalues: %i' % nconv)
    Us, Ss, Vs = getU(Es, X_p, nsv)
    return Us, Ss, Vs


def sc_svd(X_p, nsv, svdtype="trlanczos"):
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
    Ss.setDimensions(nsv=nsv, ncv=2 * nsv)
    Ss.solve()
    Vs, Us = Ss.getBV()  # fast
    return Us, Ss, Vs


# def my_svd(X_n, nsv, svdtype):
#     """
#     https://lists.mcs.anl.gov/pipermail/petsc-users/2012-May/013379.html
#     https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.732.5862&rep=rep1&type=pdf
#     """

#     # t3 = timeit.default_timer()
#     # r, c, Un = Us.createMat().getValuesCSR()
#     # r, c, Vn = Vs.createMat().getValuesCSR()
#     # Un.shape = (k, k)
#     # Vn.shape = (k, k)
#     # Sn = np.zeros(l, )
#     # for col in range(r):
#     #     Sn[col] = Ss.getValue(col)
#     # t4 = timeit.default_timer()
#     # print("PETSc to numpy {:.4f}".format(t4-t3))

#     # Sn = np.zeros(l, )
#     # for col in range(k):
#     #     Sn[col] = Ss.getValue(col)

#     # t3 = timeit.default_timer()
#     # Un = np.zeros((k, l))
#     # Sn = np.zeros(l, )
#     # Vn = np.zeros((l, k))
#     # Us_mat = Us.createMat()
#     # Vs_mat = Vs.createMat()
#     # for col in range(k):
#     #     # Vs.getColumn crashes sometimes
#     #     Un[:, col] = Us_mat.getColumnVector(col).array
#     #     Sn[col] = Ss.getValue(col)
#     #     Vn[:, col] = Vs_mat.getColumnVector(col).array
#     # t4 = timeit.default_timer()
#     # print("PETSc to numpy {:.4f}".format(t4-t3))

#     # print(np.allclose(Un0, Un))
#     # print(np.allclose(Vn0, Vn))
#     # print(np.allclose(Un1, Un))
#     # print(np.allclose(Vn1, Vn))
#     return Un, Sn, Vn.T

#     # return Us, Ss, VTs


def p2n(Ap):
    shape = Ap.getSize()
    An = np.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            An[row, col] = Ap.getValue(row, col)
    return An
# PETSc.Vec().createWithArray(array)
# PETSc.Mat().createAIJ(size=shape)
# mat.getValuesCSR()


# numpy version
# N = 1000
# X_n = np.random.rand(N, N)
def main():
    m, n, r = 12060, 5000, 1800
    # m, n, r = 12060, 500, 200
    X_n = X[:m, :n]
    # del X
    print("m, n, r: ", m, n, r)

    t0 = timeit.default_timer()
    U1, S1, VT1 = np_svd(X_n)
    dt1 = timeit.default_timer() - t0
    print(dt1)
    X_approx = np.matmul(U1[:, :r] * S1[:r], VT1[:r, :])
    error = np.std(X_approx - X_n)
    # SLEPc.SVD.Type.CYCLIC,  # slow
    # SLEPc.SVD.Type.CROSS,  # fails
    # [0] BVCreateMat() line 1408 in /home/conda/feedstock_root/build_artifacts/slepc_1578940221889/work/src/sys/classes/bv/interface/bvbasic.c
    # [0] MatCreateDense() line 1578 in /home/conda/feedstock_root/build_artifacts/petsc_1580931559030/work/src/mat/impls/dense/mpi/mpidense.c
    # [0] MatCreate() line 83 in /home/conda/feedstock_root/build_artifacts/petsc_1580931559030/work/src/mat/utils/gcreate.c
    # [0] PetscHeaderCreate_Private() line 64 in /home/conda/feedstock_root/build_artifacts/petsc_1580931559030/work/src/sys/objects/inherit.c
    # [0] PetscCommDuplicate() line 118 in /home/conda/feedstock_root/build_artifacts/petsc_1580931559030/work/src/sys/objects/tagm.c

    print("SVD with numpy took {:.4f} s, error: {:.6f}\n".format(dt1, error))
    # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # *
    t0 = timeit.default_timer()
    X_p = np2pc(X_n)
    t1 = timeit.default_timer()
    print("numpy to PETSc {:.4f}".format(t1 - t0))
    Us, Ss, Vs = sc_svd2(X_p, r)
    t2 = timeit.default_timer()
    print("SVD via EPS {:.4f}".format(t2 - t1))
    Un, Sn, Vn = BV2np(Us, Ss, Vs, r)
    t3 = timeit.default_timer()
    print("PETSc to numpy {:.4f}".format(t3 - t2))
    for svdtype in [SLEPc.SVD.Type.LAPACK,
                    SLEPc.SVD.Type.TRLANCZOS, SLEPc.SVD.Type.LANCZOS,
                    SLEPc.SVD.Type.CYCLIC, SLEPc.SVD.Type.CROSS
                    ]:
        # print(svdtype)
        t1 = timeit.default_timer()
        Us, Ss, Vs = sc_svd(X_p, r)
        t2 = timeit.default_timer()
        print(svdtype + " SVD {:.4f}".format(t2 - t1))
        print(Us.getSizes())
        print(Vs.getSizes())
        # TODO: compute error in PETSC

        # # U2, S2, VT2 = my_svd(X_n, r, svdtype)
        # # t2 = timeit.default_timer()
        # # X_approx = np.matmul(U2 * S2, VT2)
        # # error = np.std(X_approx - X_n)
        # print("SVD with SLEPc took {:.4f} s, error: {:.8f}".format(
        #     t2 - t1, error))
        # # print(np.allclose(np.abs(U1[:, :r]), np.abs(U2)))
        # # print(np.allclose(S1[:r], S2, atol=1e-10))
        # # print(np.allclose(np.abs(VT1[:r, :]), np.abs(VT2)))
        # fig, ax = plt.subplots()
        # ax.plot(S1, "g.")
        # ax.plot(S2, "r.")
        # plt.yscale("log")
        # # for bv in range(1000):
        # #     print(np.allclose(np.abs(U1[:, bv]), np.abs(U2[:, bv])), end=" ")
        # print()
# load X
# m, n, r:  12060 5000 1800
# 64.24268253520131
# SVD with numpy took 64.2427 s, error: 0.000003

# numpy to PETSc 2.2882
# Number of converged eigenvalues: 1842
# SVD via EPS 143.8802
# PETSc to numpy 0.8802
# lapack SVD 206.9961
# ((12060, 12060), 3601)
# ((5000, 5000), 3601)
# trlanczos SVD 208.1839
# ((12060, 12060), 3601)
# ((5000, 5000), 3601)
# lanczos SVD 222.0083
# ((12060, 12060), 3601)
# ((5000, 5000), 3601)
# cyclic SVD 229.8033
# ((12060, 12060), 3601)
# ((5000, 5000), 3601)
# cross SVD 216.9778
# ((12060, 12060), 3601)
# ((5000, 5000), 3601)
    # SVD with numpy took 69.1781 s, error: 0.00000345
    # reduced rank: 1800
    # lapack
    # numpy to PETSc 2.2925
    # SVD 97.5398
    # (1800, 5000, 0)
    # ((12060, 12060), 5000)
    # ((5000, 5000), 5000)
    # PETSc to numpy 1.3817
    # SVD with SLEPc took 101.8445 s, error: 0.00000345

    # trlanczos
    # numpy to PETSc 2.2738
    # SVD 208.8590
    # (1800, 3600, 3600)
    # ((12060, 12060), 3601)
    # ((5000, 5000), 3601)
    # PETSc to numpy 1.0807
    # SVD with SLEPc took 212.4911 s, error: 0.00000345

    # lanczos
    # numpy to PETSc 2.3323
    # SVD 210.6373
    # (1800, 3600, 3600)
    # ((12060, 12060), 3601)
    # ((5000, 5000), 3601)
    # PETSc to numpy 1.0814
    # SVD with SLEPc took 214.3179 s, error: 0.00000345

    # cyclic
    # numpy to PETSc 2.3363
    # SVD 720.7891
    # (1800, 3600, 3600)
    # ((12060, 12060), 3600)
    # ((5000, 5000), 3600)
    # PETSc to numpy 1.1181
    # SVD with SLEPc took 724.5172 s, error: 2585.06303611

    # cross
    # numpy to PETSc 2.3019
    # SVD 169.5902
    # (1800, 3600, 3600)
    # ((-1, -1), 0)
    # ((5000, 5000), 3600)
    # Traceback (most recent call last):

    # SVD with numpy took 70.5706 s, error: 0.00000000

    # lapack
    # numpy to PETSc 2.2711
    # SVD 102.6324
    # (1800, 5000, 0)
    # ((12060, 12060), 5000)
    # ((5000, 5000), 5000)
    # PETSc to numpy 1.3595
    # SVD with SLEPc took 106.7083 s, error: 0.00000345

    # trlanczos
    # numpy to PETSc 2.3419
    # SVD 116.2503
    # (1800, 2300, 500)
    # ((12060, 12060), 2301)
    # ((5000, 5000), 2301)
    # PETSc to numpy 0.9408
    # SVD with SLEPc took 119.7164 s, error: 0.00000345

    # lanczos
    # numpy to PETSc 2.3337
    # SVD 173.1610
    # (1800, 2300, 500)
    # ((12060, 12060), 2301)
    # ((5000, 5000), 2301)
    # PETSc to numpy 0.8984
    # SVD with SLEPc took 176.5862 s, error: 0.00000345

    # SVD with numpy took 6.6211 s, error: 0.00000000

    # lapack
    # numpy to PETSc 0.5590
    # SVD 14.2567
    # (800, 1234, 0)
    # ((12000, 12000), 1234)
    # ((1234, 1234), 1234)
    # PETSc to numpy 0.2285
    # SVD with SLEPc took 15.3814 s, error: 0.00000339

    # trlanczos
    # numpy to PETSc 0.6234
    # SVD 19.6165
    # (800, 1234, 500)
    # ((12000, 12000), 1235)
    # ((1234, 1234), 1235)
    # PETSc to numpy 0.2318
    # SVD with SLEPc took 20.5205 s, error: 0.00000339

    # lanczos
    # numpy to PETSc 0.5877
    # SVD 22.5303
    # (800, 1234, 500)
    # ((12000, 12000), 1235)
    # ((1234, 1234), 1235)
    # PETSc to numpy 0.2220
    # SVD with SLEPc took 23.3862 s, error: 0.00000339

    # SVD with numpy took 23.7183 s

    # numpy to PETSc 0.4666
    # SVD 33.9849
    # getBV 0.0018
    # PETSc to numpy 4.6777
    # SVD with SLEPc took 39.4022 s

    # SVD with numpy took 25.6435 s
    # numpy to PETSc 0.4704
    # SVD 27.9072
    # getBV 0.0001
    # PETSc to numpy 1.0757
    # SVD with SLEPc took 29.5998 s

    # print(np.allclose(np.abs(U1), np.abs(U2)))
    # print(np.allclose(S1, S2, atol=1e-10))
    # print(np.allclose(np.abs(VT1), np.abs(VT2)))
    # fig, ax = plt.subplots()
    # ax.plot(S1, "g.")
    # ax.plot(S2, "r.")
    # plt.yscale("log")

    # plt.show()
    # print('Singular values: ', S1)
    # print('Singular values: ', S2)
    # print(V)
    # print(U)

    # c0 = Vs.getColumn(0)
    # print(Vs.view())
    # print(c0.size)
    # V__ = np.array([1., 0., 0., 0., 0., 0., -0., -0., -0., -0., -0., -1., 0., 0.,
    #                 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
    #                 0., 1., 0., 0., 0., 0.]).reshape(6, 6)

    # 1000 x 1000 matrix:
    # SVD with numpy took 1.4057 s
    # numpy to PETSc took 63.0664 s
    # SVD with SLEPc took 0.5358 s
    # SLEPc to numpy took 0.0744 s
    # True
    # True
    # True


# def SVD_slpc(A):
#     # SLEPc version
#     Ap = PETSc.Mat()
#     Ap.create()
#     Ap.setSizes(A.shape)
#     Ap.setUp()
#     for row in range(A.shape[0]):
#         for col in range(A.shape[1]):
#             Ap.setValue(row, col, A[row, col])
#     Ap.assemble()

#     for stype in [SLEPc.SVD.Type.CYCLIC, SLEPc.SVD.Type.LANCZOS, SLEPc.SVD.Type.LAPACK, SLEPc.SVD.Type.TRLANCZOS]:
#         S = SLEPc.SVD()
#         S.create()
#         S.setOperator(Ap)
#         S.setType(stype)
#         S.setDimensions(A.shape[0])
#         S.solve()
#         Vs, Us = S.getBV()
#         Vs.view()
#         # Vp, _ = V.getMatrix()
#         # Up, _ = V.getMatrix()
#         s_slepc = []
#         i = 0
#         while i < S.getConverged():
#             s_slepc.append(S.getValue(i))
#             i += 1

#         print('Singular values (SLEPc {}): ['.format(S.getType()), end="")
#         for elem in s_slepc:
#             print("{:.4f}".format(elem), end=", ")
#         print("]")
#         print("{:.4f}".format(elem), end=", ")
#         print("]")


# def s2n2(Xs):
#     shape, x = Xs.getSizes()
#     Xs_mat = Xs.createMat()
#     Xn = np.zeros(shape)
#     for col in range(shape[1]):
#         # Vs.getColumn crashes sometimes
#         Xn[:, col] = Xs_mat.getColumnVector(col).array
#     return Xn
if __name__ == "__main__":
    main()
