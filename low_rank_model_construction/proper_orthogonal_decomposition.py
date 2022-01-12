# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:20:04 2021

@author: florianma
"""
import numpy as np
# from nirom.src.cross_validation import load_snapshots_cavity, plot_snapshot_cav
# from ROM.snapshot_manager import Data, load_snapshots_cavity
# import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
import warnings
plot_width = 16
LINALG_LIB = "numpy"
timed = True


def tf_svd(X, full_matrices=False):
    S, U, V = tf.linalg.svd(X, full_matrices=full_matrices)
    return U, S, transpose(V)


def np_svd(X, timed=False, full_matrices=False):
    if timed:
        tic = timeit.default_timer()
        U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
        toc = timeit.default_timer()
        print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
            X.shape[0], X.shape[1], toc-tic))
    else:
        U, S, Vh = np.linalg.svd(X, full_matrices=full_matrices)
    return U, S, Vh


if LINALG_LIB == "tensorflow":
    qr = tf.linalg.qr
    transpose = tf.transpose
    svd = tf_svd
    matmul = tf.matmul
    reshape = tf.reshape
elif LINALG_LIB == "numpy":
    qr = np.linalg.qr
    transpose = np.transpose
    svd = np_svd
    matmul = np.matmul
    reshape = np.reshape


# def timed_matmul(a, b):
#     tic = timeit.default_timer()
#     if LINALG_LIB == "tensorflow":
#         c = tf.matmul(a, b)
#     elif LINALG_LIB == "numpy":
#         c = np.matmul(a, b)
#     toc = timeit.default_timer()
#     print("matmul of \t{:.0f}\t{:.0f}\t with \t{:.0f}\t{:.0f}\t took\t{:.4}\t seconds.".format(
#         a.shape[0], a.shape[1], b.shape[0], b.shape[1], toc-tic))
#     return c


# def timed_qr(U0):
#     tic = timeit.default_timer()
#     q, r = qr(U0)
#     if LINALG_LIB == "tensorflow":
#         q, r = tf.qr(U0)
#     elif LINALG_LIB == "numpy":
#         q, r = np.qr(U0)
#     toc = timeit.default_timer()
#     print("qr decomposition of U0 \t{:.0f}\t{:.0f}\t took \t{:.4}\t seconds.".format(
#         U0.shape[0], U0.shape[1], toc-tic))
#     return q, r


# def timed_svd(X, full_matrices=False):
#     tic = timeit.default_timer()
#     if LINALG_LIB == "tensorflow":
#         U, S, VT = tf_svd(X, full_matrices=full_matrices)
#     elif LINALG_LIB == "numpy":
#         U, S, VT = np_svd(X, full_matrices=full_matrices)
#     toc = timeit.default_timer()
#     print("SVD of X \t{:.0f}\t{:.0f}\t  took \t{:.4}\t seconds.".format(
#         X.shape[0], X.shape[1], toc-tic))
#     return U, S, VT

# # if timed:
# #     qr = timed_qr
# #     svd = timed_svd
# #     matmul = timed_matmul


def plotS(S, eps):
    cum_en = np.cumsum(S)/np.sum(S)
    r = np.sum(cum_en < eps)
    n = len(S)
    fig, axs = plt.subplots(2, 1, sharex=True,
                            figsize=(plot_width/2.54, 10/2.54))
    # for i in range(3):
    axs[0].plot(np.arange(n), S, "r.")
    axs[0].plot(np.arange(r), S[:r], "g.")
    axs[1].plot(np.arange(n), cum_en, "r.")
    axs[1].plot(np.arange(r), cum_en[:r], "g.")
    axs[0].set_yscale('log')
    # axs[0].legend()
    # axs[1].legend()
    axs[0].set_title("First n singular values S")
    axs[1].set_title("Cumulative energy [%]")
    axs[0].set_xlim(0, n)
    # axs[0].set_ylim(1, 1000)
    # axs[0].set_ylim(0, S[1])
    # axs[0].set_ylim(bottom=0)
    axs[1].set_xlabel("Snapshot number")
    axs[0].set_ylabel("Singular value")
    axs[1].set_ylabel("Energy in %")
    # axs[1].set_ylim([0, 100])
    plt.tight_layout()
    plt.show()
    return


# def extend_svd(U, S, VT, X, X_additional):
#     X0 = X_additional - matmul(U, matmul(transpose(U), X_additional))
#     q = X_additional / np.linalg.norm(X_additional, axis=0)


# def hierarchical_svd(X, d, c, eps):
#     m, n = X.shape
#     Nd = np.ceil(m/d).astype(np.int32)
#     l_V, l_S = [None for i in range(d)], [None for i in range(d)]
#     for j in range(d):
#         print("slice", j, "---------------------------")
#         s, e = j*Nd, Nd*(j+1)
#         if e > m:
#             e = m
#         # print("slice", j)
#         Uc, Sc = row_svd(X[s:e, :], c, eps)
#         SVT = matmul(transpose(Uc), X[s:e, :])
#         U, S, VT = truncate_basis(*svd(SVT), eps=eps)
#         print(np.allclose(Sc, S))
#         # TODO: check if thats the same as tf.linalg.svd(X[s:e, :], full_matrices=False)
#         # print(np.allclose(VTc, VT))
#         l_S[j], U, l_V[j] = S, U, VT
#     # U, S = merge_blocks(l_V, l_S, eps)  # does not make sense
#     return U, S, VT


def row_svd(X, c, eps, ommit_V, QR_DECOMPOSITION):
    """
    Blockwise Singular Value Decomposition.

    The input matrix [X] shaped (m, n) is split into c horizontally adjacent
    blocks [X1, X2, ..., Xc] shaped (m, n1), (m, n2), ...  (m, nc).
    SVD is carried out blockwise. After truncating, the SVDs are merged
    pairwise in a hierarchical manner, until only 1 set is left.

    Parameters
    ----------
    X : (m, n) array_like
        snapshot matrix.
    c : int
        number of blocks the inpout matrix is split into. Ideally 2**k
    eps : float
        for truncating the rank of the SVDs based on the energy of the singular
        values.
    ommit_V : bool
        if True, VT is ommited while merging and calculated in the end.
    QR_DECOMPOSITION : bool
        if True, the merging algorithm will carry out a qr decomposition to
        ptentially speed up the process.

    Returns
    -------
    U : (m, r) array_like
        left singular values.
    S : (r,) array_like
        singular values.
    VT : (r, n) array_like
        right singular values.

    """
    m, n = X.shape
    Nc = np.ceil(n/c).astype(np.int32)
    t0 = timeit.default_timer()
    r = 0
    # there might not be enough data for c blocks
    c = np.ceil(n/Nc).astype(np.int32)
    additional_svds = np.ceil(np.log2(c)).astype(np.int32)
    if not isinstance(eps, str):
        eps_per_lvl = eps**(1/(1+additional_svds))
    else:
        eps_per_lvl = eps
    l_U = [None for i in range(c)]
    l_S = [None for i in range(c)]
    l_V = [None for i in range(c)]
    for j in range(c):
        print(j)
        s, e = j*Nc, Nc*(j+1)
        if e > n:
            e = n
        U, S, VT = truncate_basis(*svd(X[:, s:e]), eps_per_lvl)
        r += len(S)
        l_U[j], l_S[j], l_V[j] = U, S, VT
    t1 = timeit.default_timer()
    print("svds took {:.4f} s, reduced rank: {:.0f}".format(t1-t0, r))
    U, S, VT = merge_row(l_U, l_S, l_V, ommit_V, QR_DECOMPOSITION, eps_per_lvl)
    if ommit_V:
        VT = matmul(transpose(U)*1/S[:, None], X)
    t2 = timeit.default_timer()
    print("merging took {:.4f} s, reduced rank: {:.0f}".format(t2-t1, len(S)))
    return U, S, VT


def merge_row(l_U, l_S, l_V, ommit_V=True, QR_DECOMPOSITION=True, eps=1-1E-6):
    """
    Merge and truncate a list of SVDs in a tree based manner.

    Assumes the input matrix [X] shaped (m, n) was split into c horizontally
    adjacent blocks [X1, X2, ..., Xc] shaped (m, n1), (m, n2), ...  (m, nc).

    Parameters
    ----------
    l_U : list
        holding c Matrices with the left singular values of each block.
    l_S : list
        holding c Matrices with the singular values of each block..
    l_V : list
        holding c Matrices with the right singular values of each block..
    ommit_V : bool
        if True, VT is ommited while merging and calculated in the end.
    QR_DECOMPOSITION : bool
        if True, the merging algorithm will carry out a qr decomposition to
        ptentially speed up the process.
    eps : float
        for truncating the rank of the SVDs carried out to merge 2 blocks.
        Truncating is based on the energy of the singular values.

    Returns
    -------
    U : (m, r) array_like
        left singular values.
    S : (r,) array_like
        singular values.
    VT : (r, n) array_like
        right singular values.

    """
    if len(l_U) == 1:
        return l_U[0], l_S[0], l_V[0]
    # hierarchical tree structure
    levels = np.ceil(np.log2(len(l_U))).astype(np.int32)
    # print("merge row blocks. r1->r2: ", end="")
    for j in range(levels):
        Ni = len(l_U)
        l_Ut, l_St, l_Vt = l_U, l_S, l_V
        # if ommit_V:
        #     l_Vt = [None for i in range(Ni)]
        Ni2 = (len(l_U)+1) // 2
        l_U = [None for i in range(Ni2)]
        l_S = [None for i in range(Ni2)]
        l_V = [None for i in range(Ni2)]
        c = 0  # iteration counter (= i/2)
        r1 = 0  # rank of the ROM after merging
        r2 = 0  # rank of the ROM before merging
        for i in range(0, Ni, 2):
            if i+1 >= Ni:  # nothing to merge
                U, S, VT = l_Ut[i], l_St[i], l_Vt[i]
                r2 += len(S)
            else:
                U, S, VT = merge_horizontally(l_Ut[i], l_St[i], l_Vt[i],
                                              l_Ut[i+1], l_St[i+1], l_Vt[i+1],
                                              ommit_V, QR_DECOMPOSITION, eps)
                r2 += len(l_St[i]) + len(l_St[i+1])
            l_U[c], l_S[c], l_V[c] = U, S, VT
            r1 += len(S)
            c += 1
    #     print("{:.0f}->{:.0f}".format(r2, r1), end=" | ")
    # print()
    return U, S, VT


def merge_row_sequentially(l_U, l_S, l_V, ommit_V=True, QR_DECOMPOSITION=True, eps=1.0-1E-6):
    """
    Merge and truncate a list of SVDs in a tree based manner.

    Assumes the input matrix [X] shaped (m, n) was split into c horizontally
    adjacent blocks [X1, X2, ..., Xc] shaped (m, n1), (m, n2), ...  (m, nc).

    Parameters
    ----------
    l_U : list
        holding c Matrices with the left singular values of each block.
    l_S : list
        holding c Matrices with the singular values of each block..
    l_V : list
        holding c Matrices with the right singular values of each block..
    ommit_V : bool
        if True, VT is ommited while merging and calculated in the end.
    QR_DECOMPOSITION : bool
        if True, the merging algorithm will carry out a qr decomposition to
        ptentially speed up the process.
    eps : float
        for truncating the rank of the SVDs carried out to merge 2 blocks.
        Truncating is based on the energy of the singular values.

    Returns
    -------
    U : (m, r) array_like
        left singular values.
    S : (r,) array_like
        singular values.
    VT : (r, n) array_like
        right singular values.

    """
    if len(l_U) == 1:
        return l_U[0], l_S[0], l_V[0]

    U, S, VT = l_U[0], l_S[0], l_V[0]
    for i in range(1, len(l_U)):
        U, S, VT = merge_horizontally(U, S, VT, l_U[i], l_S[i], [i],
                                      ommit_V, QR_DECOMPOSITION, eps)
    return U, S, VT


def merge_horizontally(U1, S1, VT1, U2, S2, VT2,
                       ommit_V=True, QR_DECOMPOSITION=True, eps=1.0-1E-6):
    """
    Merge two SVD's of horizontally adjacent blocks.

    The snapshot matrix [X] shaped (m, n) is split into horizontally adjacent
    blocks [X1, X2] shaped (m, n1) and (m, n2). SVD was carried out
    individually: [X] = [X1, X2] = [U1*S1@VT1, U2*S2@VT2] = [U*S@VT].

    Parameters
    ----------
    U1 : array_like, shape (m, k)
        DESCRIPTION.
    S1 : array_like, shape (k,)
        DESCRIPTION.
    VT1 : array_like, shape (k, n1)
        DESCRIPTION.
    U2 : array_like, shape (m, l)
        DESCRIPTION.
    S2 : array_like, shape (l,)
        DESCRIPTION.
    VT2 : array_like, shape (l, n2)
        DESCRIPTION.
    eps : float, optional
        Used for truncating the SVD of . The default is 1.0.
    QR_DECOMPOSITION : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    U : array_like, shape (m, r)
        DESCRIPTION.
    S : array_like, shape (r,)
        DESCRIPTION.
    V : array_like, shape (r, n)
        might be None, can be computed later.
    """
    m, k = U1.shape
    m, l = U2.shape
    # for elem in S1[-5:]:
    #     print(S1[i], S2[i])
    # print("last 5 elements in S1")
    # print(S1[-5:])
    # print("last 5 basis vectors in U1")
    # print(U1[:, -5:])
    t1 = timeit.default_timer()
    if QR_DECOMPOSITION:  # preferred methods, as it should be usually faster
        if not isinstance(eps, str):
            if eps > 1-1E-6:
                print(eps)
                warnings.warn(
                    "there might be large numerical errors for eps>1-1E-6. Please set QR_DECOMPOSITION=False or eps=1-1e-6")
        E = np.zeros((k+l, k+l))
        U1TU2 = matmul(transpose(U1), U2)
        U0 = U2 - matmul(U1, U1TU2)  # m, l
        Q, R = qr(U0)
        E[:k, :k] = np.diag(S1)
        E[:k, k:] = U1TU2*reshape(S2, (1, l))
        E[k:, k:] = R*reshape(S2, (1, l))
        UE, S, VTE = truncate_basis(*svd(E), eps=eps)
        U = matmul(U1, UE[:k, :]) + matmul(Q, UE[k:, :])
    else:
        E = np.zeros((m, k+l))
        E[:, :k] = U1*reshape(S1, (1, k))
        E[:, k:] = U2*reshape(S2, (1, l))
        U, S, VTE = truncate_basis(*svd(E), eps=eps)
    r = len(S)
    if ommit_V:
        VT = None
    else:
        k, n1 = VT1.shape
        l, n2 = VT2.shape
        VT = np.empty((r, n1+n2))
        VT[:k, :n1] = matmul(VTE[:k, :k], VT1)
        VT[k:, :n1] = matmul(VTE[k:, :k], VT1)
        VT[:k, n1:] = matmul(VTE[:k, k:], VT2)
        VT[k:, n1:] = matmul(VTE[k:, k:], VT2)
    dt = timeit.default_timer()-t1
    print("merging {:.0f} + {:.0f} -> {:.0f} took {:.2f}".format(k, l, r, dt))
    return U, S, VT


def truncate_basis(U, S, VT, eps):
    """
    Truncate the rank of the SVD.

    Parameters
    ----------
    U : (m, n) array_like
        left singular values.
    S : (n,) array_like
        singular values.
    VT : (n, n) array_like
        right singular values.
    eps : float
        for truncating the rank of the SVDs based on the energy of the singular
        values.

    Returns
    -------
    U : (m, r) array_like
        left singular values.
    S : (r,) array_like
        singular values.
    VT : (r, n) array_like
        right singular values.

    """
    if isinstance(eps, str):
        r = int(eps[:-5])
    elif eps >= 1.0:
        # print("no truncation.")
        return U, S, VT
    elif np.sum(S) == 0:
        r = 1
    else:
        cum_en = np.cumsum(S)
        cum_en /= cum_en[-1]
        r = np.sum(cum_en <= eps)
    if r <= 1:
        r = 1
    U_hat = U[:, :r]  # (n, r)
    S_hat = S[:r]  # (r, r) / (r,) wo  non zero elements
    VT_hat = VT[:r, :]  # (r, d1)
    # plotS(S, eps)
    # print(len(S), "->", r, end="| ")
    return U_hat, S_hat, VT_hat


def merge_column(l_S, l_V, eps):
    # FIXME: repetitive. almost the same as merge_row
    levels = int(np.log2(len(l_V)))
    for j in range(levels):
        Ni = len(l_V)
        l_Vt, l_St = l_V, l_S
        Ni2 = (len(l_V)+1) // 2
        l_V, l_S = [None for i in range(Ni2)], [None for i in range(Ni2)]
        for i in range(0, Ni, 2):
            if i+1 >= Ni:
                S, VT = l_St[i], l_Vt[i]
            else:
                S, VT = merge_vertically(None, l_St[i], l_Vt[i],
                                         None, l_St[i+1], l_Vt[i+1], eps)
            l_S[i], l_V[i] = S, VT
    return U, S, VT


def merge_vertically(U1, S1, VT1, U2, S2, VT2, eps):
    # TODO: is that possible using a qr decomposition?
    # merging two bocks, [[X1], [X2]] = [X]
    k, n = VT1.shape
    l, n = VT2.shape
    E = np.zeros((k+l, n))  # n, n
    E[:k, :] = VT1*reshape(S1, (k, 1))
    E[k:, :] = VT2*reshape(S2, (l, 1))
    UE, S, VT = truncate_basis(*svd(E), eps=eps)
    if isinstance(U1, np.ndarray) | isinstance(U1, tf.Tensor):
        m1, k = U1.shape
        m2, l = U2.shape
        r = len(S)
        U = np.empty((m1+m2, r))
        U[:m1] = matmul(U1, UE[:k])
        U[m1:] = matmul(U2, UE[k:])
        return U, S, VT
    else:
        return None, S, VT


def plot_eps_vs_err(my_data):
    # all datasets
    n = 1
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1, my_data.d2)
    fig, ax = plt.subplots()
    for i in range(my_data.d2):
        print("set {:.0f}".format(i))
        X = my_data.X_n[:, ::n, i]
        f_name = "_{:03.0f}(every {:.0f} th SS).npy".format(i, n)
        U = np.load("U"+f_name)
        S = np.load("S"+f_name)
        VT = transpose(np.load("V"+f_name))

        epss = np.linspace(.98, 1.0, 21)
        err = np.zeros_like(epss)
        for j, eps in enumerate(epss):
            U2, S2, UT2 = truncate_basis(U, S, VT, eps)
            X_approx = matmul(U2, matmul(transpose(U2), X))
            err[j] = np.std(X-X_approx)
            print(eps, len(S2), err[j], sep="\t")
        plt.plot(epss, err, linestyle='-', marker='.',
                 label="set {:.0f}".format(i))
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.show()
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1*my_data.d2)
    return


def plot_eps_vs_err2(my_data):
    # 1 dataset
    # TODO: scramble ss
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1, my_data.d2)
    fig, ax = plt.subplots()
    i = 4
    print("set {:.0f}".format(i))
    X = my_data.X_n[:, :, i]
    a = np.arange(len(X[0]))
    np.random.shuffle(a)
    X = X[:, a].copy()
    # f_name = "_{:03.0f}(every {:.0f} th SS).npy".format(i, n)
    # U = np.load("U"+f_name)
    # S = np.load("S"+f_name)
    # VT = transpose(np.load("V"+f_name))

    epss = np.linspace(.98, 1.0, 21)
    err = np.zeros_like(epss)
    for j, eps in enumerate(epss):
        # U2, S2, UT2 = truncate_basis(U, S, VT, eps)
        U2, S2, VT2 = row_svd(X, 4, eps=eps, ommit_V=True,
                              QR_DECOMPOSITION=True)
        X_approx = matmul(U2, matmul(transpose(U2), X))
        err[j] = np.std(X-X_approx)
        print(eps, len(S2), err[j], sep="\t")
    plt.plot(epss, err, linestyle='-', marker='.',
             label="set {:.0f}".format(i))
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.show()
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1*my_data.d2)
    return


def plot_eps_vs_err3(my_data):
    # 1 dataset
    # TODO: scramble ss
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1, my_data.d2)
    fig, ax = plt.subplots()
    i = 4
    print("set {:.0f}".format(i))
    X = my_data.X_n[:, :, i]
    a = np.arange(len(X[0]))
    np.random.shuffle(a)
    X = X[:, a].copy()
    # f_name = "_{:03.0f}(every {:.0f} th SS).npy".format(i, n)
    # U = np.load("U"+f_name)
    # S = np.load("S"+f_name)
    # VT = transpose(np.load("V"+f_name))

    epss = 1-1/10**np.arange(0, 18)
    err = np.zeros_like(epss)
    for j, eps in enumerate(epss):
        # U2, S2, UT2 = truncate_basis(U, S, VT, eps)
        U2, S2, VT2 = row_svd(X, 4, eps=eps, ommit_V=True,
                              QR_DECOMPOSITION=True)
        X_approx = matmul(U2, matmul(transpose(U2), X))
        err[j] = np.std(X-X_approx)
        print(eps, len(S2), err[j], sep="\t")
    plt.plot(epss, err, linestyle='-', marker='.',
             label="set {:.0f}".format(i))
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("error")
    plt.show()
    my_data.X_n.shape = (my_data.s1*my_data.s2, my_data.d1*my_data.d2)
    return

# np.lib.index_tricks.nd_grid()


# class POD(Data):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     # def svd(self, X):
#     #     S, U, VT = tf.linalg.svd(X, full_matrices=False)
#     #     return U.numpy(), S.numpy(), V.numpy()

#     def hierarchical_pod(self, eps):
#         """
#         a tree based merge-and-truncate algorithm to obtain an approximate
#         truncated SVD of the matrix.

#         Parameters
#         ----------
#         eps : float
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         # Hierarchical Singular Value Decomposition
#         return

#     def higher_order_svd(a, full_matrices=True):
#         # a = np.random.rand(2, 3, 4, 5)
#         # U, S = higher_order_svd(a)
#         # print(S.shape)
#         # print([i.shape for i in U])
#         # a1 = S
#         # for i, _ in enumerate(U):
#         # a1 = np.tensordot(a1, U[i], (0, 1))
#         # print(np.allclose(a, a1))
#         # core_tensor = a
#         left_singular_basis = []
#         for k in range(a.ndim):
#             unfold = np.reshape(np.moveaxis(a, k, 0), (a.shape[k], -1))
#             U, _, _ = svd(unfold, full_matrices=full_matrices)
#             left_singular_basis.append(U)
#             U_c = transpose(U).conj()
#             a = np.tensordot(a, U_c, (0, 1))
#         return left_singular_basis, a

#     def _2_way_pod(self, eps1, eps2):
#         # #[eps, error, n basis vectors (avg, per dataset)],
#         # res = np.array([
#         # [1.000000000, 0.000649143, 416.3], [1, 0.000649143, 416.3],
#         # [0.994897959, 0.001054028, 251.5], [0.989795918, 0.00118231, 211],
#         # [0.984693878, 0.001342111, 181.3], [0.979591837, 0.001533657, 157.4],
#         # [0.974489796, 0.001778205, 137.4], [0.973684211, 0.001824512, 134.5],
#         # [0.969387755, 0.002076038, 120.3], [0.964285714, 0.002402765, 105.2],
#         # [0.959183673, 0.002712533, 91.7], [0.954081633, 0.003031809, 79.9],
#         # [0.948979592, 0.003378348, 69.5], [0.947368421, 0.00349487, 66.3],
#         # [0.943877551, 0.003744284, 60.2], [0.93877551, 0.004137558, 52],
#         # [0.933673469, 0.004531132, 45.2], [0.928571429, 0.004951166, 39.3],
#         # [0.923469388, 0.005380647, 34.4], [0.921052632, 0.005613473, 32.2],
#         # [0.918367347, 0.005870244, 30.1], [0.913265306, 0.006364088, 26.4],
#         # [0.908163265, 0.006875367, 23.5], [0.903061224, 0.007418286, 20.9],
#         # [0.897959184, 0.008020077, 18.6], [0.894736842, 0.00837866, 17.4],
#         # [0.892857143, 0.008667827, 16.6], [0.887755102, 0.009338536, 15],
#         # [0.882653061, 0.010072033, 13.6], [0.87755102, 0.010745011, 12.4],
#         # [0.872448980, 0.011194270, 11.6], [0.868421053, 0.01183017, 10.8],
#         # [0.867346939, 0.012081178, 10.5], [0.862244898, 0.012796338, 9.8],
#         # [0.857142857, 0.013526181, 9.2], [0.852040816, 0.014283143, 8.5],
#         # [0.846938776, 0.015037375, 8.0], [0.842105263, 0.015727348, 7.7],
#         # [0.841836735, 0.015727348, 7.7], [0.836734694, 0.016882615, 7.1],
#         # [0.831632653, 0.017513710, 6.7], [0.826530612, 0.018314773, 6.4],
#         # [0.821428571, 0.018648107, 6.2], [0.815789474, 0.020361032, 5.7],
#         # [0.816326531, 0.020361032, 5.7], [0.81122449, 0.02121174, 5.5],
#         # [0.806122449, 0.022170866, 5.3], [0.801020408, 0.022990563, 5.2],
#         # [0.795918367, 0.023983465, 5.0], [0.789473684, 0.027309363, 4.4],
#         # [0.790816327, 0.027309363, 4.4], [0.785714286, 0.027996551, 4.3],
#         # [0.780612245, 0.029564418, 4.2], [0.775510204, 0.031077579, 4.1],
#         # [0.765306122, 0.032118860, 4.0], [0.770408163, 0.03211886, 4],
#         # [0.763157895, 0.032865164, 3.7], [0.760204082, 0.033701762, 3.4],
#         # [0.755102041, 0.034560932, 3.3], [0.75, 0.035673262, 3.2],
#         # [0.736842105, 0.038180729, 3.0], [0.710526316, 0.042435443, 2.3],
#         # [0.684210526, 0.062167254, 2.0], [0.657894737, 0.064424002, 1.9],
#         # [0.578947368, 0.135025099, 1.0], [0.605263158, 0.135025099, 1],
#         # [0.631578947, 0.135025099, 1.0], [0.552631579, 0.137783313, 0.9],
#         # [0.526315789, 0.139093005, 0.8], [0.5, 0.15059241, 0.5]])
#         self.X_n.shape = (self.s1*self.s2, self.d1, self.d2)
#         U_hats = np.zeros((self.s1*self.s2, self.d1*self.d2))
#         s, e = 0, 0
#         n = 1
#         for i in range(self.d2):
#             X = self.X_n[:, :: n, i]
#             f_name = "_{:03.0f}(every {:.0f} th SS).npy".format(i, n)
#             try:
#                 path = "C:/Users/florianma/"
#                 U = np.load(path+"U"+f_name)
#                 S = np.load(path+"S"+f_name)
#                 VT = transpose(np.load(path+"V"+f_name))

#                 # T = self.xi[i, 1]
#                 # path = "C:/Users/florianma/Documents/data/freezing_cavity/"
#                 # np.save(path+"Tamb{:.0f}_{}.npy".format(T, "U"), U)
#                 # np.save(path+"Tamb{:.0f}_{}.npy".format(T, "S"), S)
#                 # np.save(path+"Tamb{:.0f}_{}.npy".format(T, "VT"), VT)
#                 # print("loaded: "+f_name)
#             except:
#                 U, S, VT = self.svd(X)
#                 np.save("U"+f_name, U)
#                 np.save("S"+f_name, S)
#                 np.save("VT"+f_name, VT)
#                 # print("saved: "+f_name)
#             S_hat, U_hat, VT_hat = self.truncate_basis(S, U, VT, eps1)
#             e = s + U_hat.shape[1]
#             U_hats[:, s: e] = U_hat
#             # print(i, X.shape, U.shape, U_hat.shape)
#             s = e
#         U_hats = U_hats[:, : e]
#         self.X_n.shape = (self.s1*self.s2, self.d1*self.d2)
#         self.U_hats = U_hats

#         # print(U_hats.shape)
#         # self = my_POD
#         # U_hats = self.U_hats
#         # U_hat = self.U_hat
#         # S, U, VT = self.S, self.U, self.VT
#         # S_hat, U_hat, VT_hat = self.S_hat, self.U_hat, self.VT_hat
#         # print(U_hats.shape)

#         U, S, VT = self.svd(U_hats)
#         S_hat, U_hat, VT_hat = self.truncate_basis(S, U, VT, eps2)
#         # print(S.shape, S_hat.shape)
#         # print(U.shape, U_hat.shape)
#         # print(VT.shape, VT_hat.shape)

#         X_n_approx = matmul(U_hat, matmul(transpose(U_hat), self.X_n))
#         error = np.std(self.X_n-X_n_approx)

#         # self.S, self.U, self.VT = S, U, VT
#         # self.S_hat, self.U_hat, self.VT_hat = S_hat, U_hat, VT_hat
#         # self.X_n_approx = X_n_approx

#         # print(error)
#         # print(np.mean(np.abs(self.X_n-X_n_approx)))
#         # print(np.mean(np.abs(self.X_n/X_n_approx)))
#         # self.X_n.shape = (self.s1*self.s2, self.d1, self.d2)
#         # X_n_approx.shape = (self.s1*self.s2, self.d1, self.d2)
#         # for i in range(60):
#         #     plot_snapshot_cav(X_n_approx[:, 500, i], x, y, tri)
#         #     plt.show()
#         #     plot_snapshot_cav(self.X_n[:, 500, i], x, y, tri)
#         #     plt.show()

#         # rel_energy = np.cumsum(S) / np.sum(S)
#         # print(rel_energy.shape)
#         # fig, ax = plt.subplots()
#         # plt.plot(rel_energy, "b.")
#         # plt.show()
#         return error

#     def truncate_basis(self, S, U, VT, eps):
#         """
#         reduce rank of basis to low rank r
#         """
#         cum_en = np.cumsum(S)/np.sum(S)
#         r = np.sum(cum_en < eps)
#         U_hat = U[:, : r]  # (n, r)
#         S_hat = S[: r]  # (r, r) / (r,) wo  non zero elements
#         VT_hat = VT[: r, :]  # (r, d1)

#         self.cum_en = cum_en
#         # n = len(S)
#         # fig, axs = plt.subplots(2, 1, sharex=True,
#         #                         figsize=(plot_width/2.54, 10/2.54))
#         # # for i in range(3):
#         # axs[0].plot(np.arange(n), S, "r.")
#         # axs[0].plot(np.arange(r), S[:r], "g.")
#         # axs[1].plot(np.arange(n), cum_en, "r.")
#         # axs[1].plot(np.arange(r), cum_en[:r], "g.")
#         # axs[0].set_yscale('log')
#         # # axs[0].legend()
#         # # axs[1].legend()
#         # axs[0].set_title("First n singular values S")
#         # axs[1].set_title("Cumulative energy [%]")
#         # axs[0].set_xlim(0, n)
#         # # axs[0].set_ylim(1, 1000)
#         # # axs[0].set_ylim(0, S[1])
#         # # axs[0].set_ylim(bottom=0)
#         # axs[1].set_xlabel("Snapshot number")
#         # axs[0].set_ylabel("Singular value")
#         # axs[1].set_ylabel("Energy in %")
#         # # axs[1].set_ylim([0, 100])
#         # plt.tight_layout()
#         # plt.show()
#         return S_hat, U_hat, VT_hat

#     def to_reduced_space(self):
#         return

#     def from_reduced_space(self):
#         return

#     # predict
#     def predict(S, U, VT, r):
#         # S_hat = S.numpy()[:r]  # (r, r) / (r,) wo  non zero elements
#         # U_hat = U.numpy()[:, :r]  # (n, r)
#         # VT_hat = VT[:, :r]  # (d1, r)
#         X_approx = matmul(U*S, VT)  # n, d1
#         return X_approx


def test_merge(N):
    eps = 1.0
    assert N > 4
    X = np.random.rand(N, N)
    U_, S_, VT_ = svd(X, False)

    print("split vertically: X = [[X1], [X2]] = ...", end=" ")
    m1, n1 = np.random.randint(1, N-1), np.random.randint(1, N-1)
    U1, S1, VT1 = svd(X[: m1, :], False)
    U2, S2, VT2 = svd(X[m1:, :], False)
    U, S, VT = merge_vertically(U1, S1, VT1, U2, S2, VT2, eps)
    assert np.allclose(np.abs(U/U_), 1), "merge_vertically failed, U differs."
    assert np.allclose(S, S_), "merge_vertically failed, S differs."
    assert np.allclose(
        np.abs(VT/VT_), 1), "merge_vertically failed, VT differs."
    print("O.K.")

    print("split horizontally: X = [X1, X2]....", end=" ")
    U1, S1, VT1 = svd(X[:, :n1], False)
    U2, S2, VT2 = svd(X[:, n1:], False)

    U, S, VT = merge_horizontally(U1, S1, VT1, U2, S2, VT2, eps)
    assert np.allclose(
        np.abs(U/U_), 1), "merge_horizontally failed, U differs."
    assert np.allclose(S, S_), "merge_horizontally failed, S differs."
    assert np.allclose(
        np.abs(VT/VT_), 1), "merge_horizontally failed, VT differs."
    print("merge_horizontally with qr decomposition O.K.")

    U, S, VT = merge_horizontally(
        U1, S1, VT1, U2, S2, VT2, eps, QR_DECOMPOSITION=False)
    assert np.allclose(
        np.abs(U/U_), 1), "merge_horizontally failed, U differs."
    assert np.allclose(S, S_), "merge_horizontally2 failed, S differs."
    assert np.allclose(
        np.abs(VT/VT_), 1), "merge_horizontally2 failed, VT differs."
    print("merge_horizontally O.K.")
    return


def test_sequential_merge(my_data):
    eps = .99999
    Ts = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625])

    additional_svds = np.ceil(np.log2(len(Ts))).astype(np.int32)
    eps_per_lvl = eps**(1/(1+additional_svds))
    l_U, l_S, l_VT = [], [], []
    t1 = timeit.default_timer()
    for i, t_amb in enumerate(Ts):  # iteration along d2
        U = np.load(path+"Tamb{:.0f}_U.npy".format(t_amb))
        S = np.load(path+"Tamb{:.0f}_S.npy".format(t_amb))
        VT = np.load(path+"Tamb{:.0f}_VT.npy".format(t_amb))
        U, S, VT = truncate_basis(U, S, VT, eps_per_lvl)
        l_U[i:i], l_S[i:i], l_VT[i:i] = [U], [S], [VT]
    U1, S1, VT1 = merge_row(l_U, l_S, l_VT, ommit_V=True,
                            QR_DECOMPOSITION=True, eps=eps)
    t2 = timeit.default_timer()
    print("merging took {:.4f} s, reduced rank: {:.0f}".format(t2-t1, len(S1)))
    X_approx = matmul(U1, matmul(transpose(U1), my_data.X_n))
    print(np.std(my_data.X_n-X_approx))
    # merge row blocks. r1->r2: eps = 1-1e-4
    # SVD of X 	5586	5586	  took 	106.6	 seconds.
    # SVD of X 	5903	5903	  took 	104.5	 seconds.
    # SVD of X 	6306	6306	  took 	118.6	 seconds.
    # SVD of X 	6907	6907	  took 	171.1	 seconds.
    # SVD of X 	7679	7679	  took 	276.0	 seconds.
    # 32381->22397 |
    # SVD of X 	8485	8485	  took 	316.5	 seconds.
    # SVD of X 	9098	9098	  took 	402.0	 seconds.
    # 22397->14852 |
    # SVD of X 	10038	10038	  took 	508.7	 seconds.
    # 14852->10221 |
    # SVD of X 	10221	10221	  took 	540.6	 seconds.
    # 10221->5596 |
    # merging took 2898.1897 s, reduced rank: 5596
    # 7.93563027082641e-06
    # 106.6+104.5+118.6+171.1+276.0 + 316.5+402.0 + 508.7 + 540.6 = 2544.6

    # eps = .999
    # SVD of X 	4858	4858	  took 	53.4	 seconds.
    # merging 2406 + 2452 -> 3056 took 63.42
    # SVD of X 	5102	5102	  took 	77.19	 seconds.
    # merging 2498 + 2604 -> 3131 took 91.04
    # SVD of X 	5381	5381	  took 	73.35	 seconds.
    # merging 2641 + 2740 -> 3226 took 88.64
    # SVD of X 	5797	5797	  took 	88.22	 seconds.
    # merging 2832 + 2965 -> 3338 took 104.81
    # SVD of X 	6224	6224	  took 	110.5	 seconds.
    # merging 3142 + 3082 -> 3359 took 127.90
    # SVD of X 	6187	6187	  took 	104.8	 seconds.
    # merging 3056 + 3131 -> 3601 took 122.81
    # SVD of X 	6564	6564	  took 	120.8	 seconds.
    # merging 3226 + 3338 -> 3827 took 139.60
    # SVD of X 	7428	7428	  took 	177.3	 seconds.
    # merging 3601 + 3827 -> 4177 took 199.86
    # SVD of X 	7536	7536	  took 	181.9	 seconds.
    # merging 4177 + 3359 -> 4374 took 204.40
    # merging took 1154.5480 s, reduced rank: 4374
    # 7.382204639301287e-05
    additional_svds = len(Ts)
    eps_per_lvl = eps**(1/(1+additional_svds))
    l_U, l_S, l_VT = [], [], []
    t1 = timeit.default_timer()
    for i, t_amb in enumerate(Ts):  # iteration along d2
        U = np.load(path+"Tamb{:.0f}_U.npy".format(t_amb))
        S = np.load(path+"Tamb{:.0f}_S.npy".format(t_amb))
        VT = np.load(path+"Tamb{:.0f}_VT.npy".format(t_amb))
        U, S, VT = truncate_basis(U, S, VT, eps_per_lvl)
        l_U[i:i], l_S[i:i], l_VT[i:i] = [U], [S], [VT]
    U2, S2, VT2 = merge_row_sequentially(l_U, l_S, l_VT, ommit_V=True,
                                         QR_DECOMPOSITION=True, eps=eps)
    t2 = timeit.default_timer()
    print("merging took {:.4f} s, reduced rank: {:.0f}".format(t2-t1, len(S2)))
    X_approx = matmul(U2, matmul(transpose(U2), my_data.X_n))
    print(np.std(my_data.X_n-X_approx))
    # merge row blocks. r1->r2: eps = 1-1e-4
    # SVD of X 	5739	5739	  took 	141.5	 seconds. 2832+2907->4190 |
    # SVD of X 	7163	7163	  took 	279.6	 seconds. 4190+2973->4661 |
    # SVD of X 	7754	7754	  took 	328.5	 seconds. 4661+3093->4844 |
    # SVD of X 	8023	8023	  took 	320.4	 seconds. 4844+3179->4962 |
    # SVD of X 	8279	8279	  took 	384.7	 seconds. 4962+3317->5062 |
    # SVD of X 	8531	8531	  took 	382.1	 seconds. 5062+3469->5158 |
    # SVD of X 	8840	8840	  took 	407.0	 seconds. 5158+3682->5256 |
    # SVD of X 	9224	9224	  took 	411.9	 seconds. 5256+3968->5369 |
    # SVD of X 	9407	9407	  took 	422.2	 seconds. 5369+4038->5470 |
    # merging took 3444.0238 s, reduced rank: 5470
    # 0.00021152030820343782
    # 141.5+279.6+328.5+320.4+384.7+382.1+407.0+411.9+422.2 = 3077.9

    # eps = .999
    # SVD of X 	5165	5165	  took 	62.79	 seconds.
    # merging 2555 + 2610 -> 3085 took 78.30
    # SVD of X 	5749	5749	  took 	82.67	 seconds.
    # merging 3085 + 2664 -> 3373 took 97.37
    # SVD of X 	6151	6151	  took 	126.4	 seconds.
    # merging 3373 + 2778 -> 3509 took 144.70
    # SVD of X 	6336	6336	  took 	147.0	 seconds.
    # merging 3509 + 2827 -> 3597 took 171.68
    # SVD of X 	6538	6538	  took 	167.9	 seconds.
    # merging 3597 + 2941 -> 3677 took 193.88
    # SVD of X 	6723	6723	  took 	156.3	 seconds.
    # merging 3677 + 3046 -> 3758 took 184.26
    # SVD of X 	6967	6967	  took 	177.4	 seconds.
    # merging 3758 + 3209 -> 3853 took 206.81
    # SVD of X 	7272	7272	  took 	187.1	 seconds.
    # merging 3853 + 3419 -> 3974 took 217.74
    # SVD of X 	7380	7380	  took 	183.5	 seconds.
    # merging 3974 + 3406 -> 4061 took 209.03
    # merging took 1513.4007 s, reduced rank: 4061
    # 0.00010160153641296128


if __name__ == "__main__":
    from ROM.snapshot_manager import Data, load_snapshots_cavity
    if "my_data" not in locals():
        path = "C:/Users/florianma/Documents/data/freezing_cavity/"
        # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
        my_data = Data(*load_snapshots_cavity(path))
    asd
    test_sequential_merge(my_data)
    asd
    X = my_data.X_n
    a = np.arange(len(X[0]))
    np.random.shuffle(a)
    # X =
    U, S, VT = row_svd(X[:, a], 16, eps=1.0-1E-5,
                       ommit_V=True, QR_DECOMPOSITION=True)
    asd
    plot_eps_vs_err3(my_data)
    X = np.random.rand(20, 3)
    U, S, VT = row_svd(X, 1, eps=1-1E-6, ommit_V=True,
                       QR_DECOMPOSITION=True)
    print(VT)
    U, S, VT = row_svd(X, 1, eps=1-1E-6, ommit_V=False,
                       QR_DECOMPOSITION=True)
    print(VT)
    U, S, VT = row_svd(X, 1, eps=1-1E-6, ommit_V=False,
                       QR_DECOMPOSITION=False)
    print(VT)
    U, S, VT = row_svd(X, 1, eps=1.0-1E-6, ommit_V=True,
                       QR_DECOMPOSITION=False)
    print(VT)
    print(U.shape, S.shape, VT.shape)
    # asd
    np.save("xi.npy", my_data.xi)

    asd
    X = my_data.X_n[:, :]
    # X_all = np.random.rand(2000, 2000)

    eps = .999
    print(eps)

    U, S, VT = row_svd(X, 16, eps, ommit_V=True,
                       QR_DECOMPOSITION=True)
    print(U.shape, S.shape, VT.shape)
    X_approx = matmul(U, matmul(transpose(U), X))
    err = np.std(X-X_approx)
    0.999
    # svds took 2866.2433 s, reduced rank: 19410
    # merge row blocks. r1->r2: 19410->13511 | 13511->9388 | 9388->6865 | 6865->5240 |
    # merging took 2635.6923 s, reduced rank: 5240
    # 2.1565296555624087e-05
    f_name = "50kSVD.npy"
    np.save("U"+f_name, U)
    np.save("S"+f_name, S)
    np.save("VT"+f_name, VT)

    asd

    if X.shape[1] < 3000:
        t0 = timeit.default_timer()
        U1, S1, VT1 = truncate_basis(*svd(X), eps)
        t1 = timeit.default_timer()
        X_approx = matmul(U1, matmul(transpose(U1), X))
        error1 = np.std(X-X_approx)
        print("SVD1: ", t1-t0, error1, U1.shape[1])
        print("-----------------------")
    # print(np.allclose(X, matmul(U1*S1, VT1)))
    # if True:
    #     n = 8
    for i in np.arange(1, 10):
        n = 2**i
        if (X.shape[1] / n < 3000) and (X.shape[1] / n > 100):
            eps = 1.0
            print(n)
            t2 = timeit.default_timer()
            U2, S2, VT2 = row_svd(X, n, eps, ommit_V=True,
                                  QR_DECOMPOSITION=True)
            t3 = timeit.default_timer()
            X_approx = matmul(U2, matmul(transpose(U2), X))
            error2 = np.std(X-X_approx)
            print("SVD: ", t3-t2, error2)

            # t2 = timeit.default_timer()
            # U, S, VT = svd(X)
            # U2, S2, VT2 = truncate_basis(U, S, VT, eps=.999)
            # U3, S3, VT3 = truncate_basis(U, S, VT, eps=1.0-1E-6)
            # t3 = timeit.default_timer()

            # X_approx = matmul(U2, matmul(transpose(U2), X))
            # error2 = np.std(X-X_approx)
            # print("SVD: ", t3-t2, error2)

            # X_approx = matmul(U, matmul(transpose(U), X))
            # error2 = np.std(X-X_approx)
            # print("SVD: ", t3-t2, error2)

            # X_approx = matmul(U3, matmul(transpose(U3), X))
            # error2 = np.std(X-X_approx)
            # print("SVD: ", t3-t2, error2)

            # print("-----------------------")

            # t3 = timeit.default_timer()
            # U3, S3, VT3 = row_svd(X, n, eps, ommit_V=True,
            #                       QR_DECOMPOSITION=False)
            # t4 = timeit.default_timer()
            # X_approx = matmul(U3, matmul(transpose(U3), X))
            # error3 = np.std(X-X_approx)
            # print("SVD3: ", t4-t3, error3)
            # print("-----------------------")

            # t2 = timeit.default_timer()
            # U2, S2, VT2 = row_svd(X, n, eps, ommit_V=False,
            #                       QR_DECOMPOSITION=True)
            # t3 = timeit.default_timer()
            # X_approx = matmul(U2, matmul(transpose(U2), X))
            # error2 = np.std(X-X_approx)
            # print("SVD4: ", t3-t2, error2)
            # print("-----------------------")

            # t3 = timeit.default_timer()
            # U3, S3, VT3 = row_svd(X, n, eps, ommit_V=False,
            #                       QR_DECOMPOSITION=False)
            # t4 = timeit.default_timer()
            # X_approx = matmul(U3, matmul(transpose(U3), X))
            # error3 = np.std(X-X_approx)
            # print("SVD5: ", t4-t3, error3)
            # print("-----------------------")
            # print("-----------------------")
            # print("-----------------------")

            # t41 = timeit.default_timer()
            # X_approx = matmul(U2, matmul(transpose(U2), X))
            # t42 = timeit.default_timer()
            # X_approx = matmul(matmul(U2, transpose(U2)), X)
            # t43 = timeit.default_timer()
            # print(t42-t41)
            # print(t43-t42)

            # print("-----------------------")
            # print(np.allclose(X, matmul(U2*S2, VT2)))
            # print(np.allclose(X, matmul(U3*S3, VT3)))

    # print(U.shape, U_.shape, np.allclose(np.abs(U/U_), 1))
    # print(S.shape, S_.shape, np.allclose(S, S_, atol=1e-5, rtol=1e-5))
    # print(VT.shape, VT_.shape, np.allclose(np.abs(VT/VT_), 1))

    # asd
    # for N in [100, 250, 500, 1000]:
    #     # N = 2000
    #     M = N
    #     m1, n1 = M//2, N//2
    #     X = X_all[:M, :N]
    #     t0 = timeit.default_timer()
    #     U_, S_, VT_ = svd(X, False)
    #     t1 = timeit.default_timer()
    #     # X = np.random.rand(1000, 1000)
    #     U_, S_, VT_ = svd(X, False)
    #     # print("split vertically: X = [[X1], [X2]]", M, N)
    #     # t2 = timeit.default_timer()
    #     # try:
    #     #     U1, S1, VT1 = svd(X[:m1, :], False)
    #     # except:
    #     #     U1, S1, VT1 = timed_svd_truncated(X[:m1, :], False)
    #     # U2, S2, VT2 = svd(X[m1:, :], False)
    #     # t3 = timeit.default_timer()
    #     # print(t1-t0)
    #     # print(t3-t2)

    #     # t1 = timeit.default_timer()
    #     # U, S, VT = merge_vertically(U1, S1, VT1, U2, S2, VT2, eps)
    #     # t2 = timeit.default_timer()
    #     # print(np.allclose(X, matmul(U*S, V)))

    #     print("split horizontally: X = [X1, X2]")
    #     U1, S1, VT1 = svd(X[:, :n1], False)
    #     U2, S2, VT2 = svd(X[:, n1:], False)

    #     t3 = timeit.default_timer()
    #     # U, S, VT = merge_horizontally(U1, S1, VT1, U2, S2, VT2, eps)
    #     t4 = timeit.default_timer()
    #     # print(np.allclose(X, matmul(U*S, VT)))
    #     t5 = timeit.default_timer()
    #     U, S, VT = merge_horizontally2(U1, S1, VT1, U2, S2, VT2, eps)
    #     t6 = timeit.default_timer()
    #     U, S = merge_horizontally2(U1, S1, None, U2, S2, None, eps)
    #     t62 = timeit.default_timer()
    #     print(t6-t5)
    #     print(t62-t5)
    #     print(np.allclose(X, matmul(U*S, VT)))

    #     # merge_blocks(l_U, l_S, eps)
    #     t7 = timeit.default_timer()
    #     U, S = merge_row([U1, U2], [S1, S2], ommit_V=False, eps=eps)
    #     t8 = timeit.default_timer()
    #     print(np.allclose(X, matmul(U*S, VT)))
    #     print(U.shape, U_.shape, np.allclose(np.abs(U/U_), 1))
    #     print(U.shape, U_.shape, np.allclose(np.abs(U), np.abs(U_)))
    #     # print(S.shape, S_.shape, np.allclose(S, S_, atol=1e-5, rtol=1e-5))
    #     print(M, N)
    #     # print("merge_vertically", t2-t1)
    #     print("merge_horizontally", t4-t3)
    #     # print("merge_horizontally2", t6-t5)
    #     print("merge_row", t8-t7)
    #     print()
    #     print()

    # X[:500, :400] = 0; X  # 1, 1
    # X[500:, :300] = 0; X  # 2, 1
    # X[:500, -200:] = 0; X  # 1, 2
    # X[500:, -100:] = 0; X  # 2, 2
    # d, c = 2, 2
    # tic = timeit.default_timer()
    # U, S, VT = hierarchical_svd(X, d, c, .9999)
    # toc = timeit.default_timer()
    # print("hierarchical svd X (shape ", X.shape, ") took {:.1f} seconds.".format(toc-tic))
    # U, S, VT = svd(X, full_matrices=False)
    # np.linalg.svd()
    # X = np.random.rand(2000, 2000)
    # # for n in range(100, 2000, 100):
    # #     matmul(X[:, :n], X[:n, :])
    # #     matmul(X[:n, :], X[:, :n])
    # #     matmul(X[:n, :n], X[:n, :n])
    # # # for n in range(100, 1000, 25):
    # for n in range(100, 2000, 100):
    #     svd(X[:n, :], False)
    #     svd(X[:, :n], False)
    #     svd(X[:n, :n], False)
    # for n in range(100, 2000, 100):
    #     timed_svd2(X[:n, :], False)
    #     timed_svd2(X[:, :n], False)
    #     timed_svd2(X[:n, :n], False)

    # # S, U, VT = tf.linalg.svd(X_all, full_matrices=False)
    # my_POD = POD(X_all, _xi_all_, x, y, tri, dims_all)
    # for eps in np.linspace(0.8, 1.0, 21):
    #     if eps == 1.0:
    #         eps = .9925
    #     error = my_POD._2_way_pod(eps, 1.0)
    #     print(eps, error, my_POD.U_hats.shape[1])
    #     error = my_POD._2_way_pod(eps, 1.0)
    #     print(eps, error, my_POD.U_hats.shape[1])
    #     print(eps, error, my_POD.U_hats.shape[1])
