# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:20:04 2021

@author: florianma
"""
import numpy as np
from nirom.src.cross_validation import load_snapshots_cavity, plot_snapshot_cav
import tensorflow as tf
import matplotlib.pyplot as plt
plot_width = 16
import timeit

def timed_dot(a, b):
    tic = timeit.default_timer()
    c = np.dot(a, b)
    toc = timeit.default_timer()
    print("dot of \t {:.0f} \t {:.0f}  \t with \t {:.0f} \t {:.0f}  \t took \t {:.4} \t seconds.".format(a.shape[0], a.shape[1], b.shape[0], b.shape[1], toc-tic))
    return c

def timed_qr(U0):
    tic = timeit.default_timer()
    q, r = np.linalg.qr(U0)
    toc = timeit.default_timer()
    # print("qr decomposition of U0 (shape ", U0.shape, ") took {:.1f} seconds.".format(toc-tic))
    print("qr decomposition of U0 \t {:.0f} \t {:.0f}  \t  took \t {:.4} \t seconds.".format(U0.shape[0], U0.shape[1], toc-tic))
    return q, r

def timed_svd(X, full_matrices):
    tic = timeit.default_timer()
    S, U, V = tf.linalg.svd(X, full_matrices=full_matrices)
    S, U, V = S.numpy(), U.numpy(), V.numpy()
    toc = timeit.default_timer()
    print("SVD of X \t {:.0f} \t {:.0f}  \t  took \t {:.4} \t seconds.".format(X.shape[0], X.shape[1], toc-tic))
    return U, S, V

svd = timed_svd
dot = timed_dot
qr = timed_qr


def plotS(S, eps):
    cum_en = np.cumsum(S)/np.sum(S)
    r = np.sum(cum_en<eps)
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

def truncate_basis(S, U, V, eps):
    """
    reduce rank of basis to low rank r
    """
    cum_en = np.cumsum(S)/np.sum(S)
    r = np.sum(cum_en<eps)
    U_hat = U[:, :r]  # (n, r)
    S_hat = S[:r]  # (r, r) / (r,) wo  non zero elements
    V_hat = V[:, :r]  # (d1, r)
    # plotS(S, eps)
    print(len(S), "->", r)
    return S_hat, U_hat, V_hat


def extend_svd(U, S, V, X, X_additional):
    X0 = X_additional - np.dot(U, np.dot(U.T, X_additional))
    q = X_additional / np.linalg.norm(X_additional, axis=0)


def hierarchical_svd(X, d, c, eps):
    m, n = X.shape
    Nd = np.ceil(m/d).astype(np.int32)
    l_V, l_S = [None for i in range(d)], [None for i in range(d)]
    for j in range(d):
        print("slice", j, "---------------------------")
        s, e = j*Nd, Nd*(j+1)
        if e>m: e=m
        # print("slice", j)
        Uc, Sc = blockwise_svd(X[s:e, :], c, eps)


        SV = dot(Uc.T, X[s:e])
        U, S, V = svd(SV, full_matrices=False)
        print(np.allclose(Sc, S))
        # TODO: check if thats the same as tf.linalg.svd(X[s:e, :], full_matrices=False)
        # print(np.allclose(Vc, V))
        l_S[j], U, l_V[j] = S, U, V
    # U, S = merge_blocks(l_V, l_S)  # does not make sense
    return U, S, V



def blockwise_svd(X, c, eps):
    m, n = X.shape
    l_U, l_S = [None for i in range(c)], [None for i in range(c)]
    Nc = np.ceil(n/c).astype(np.int32)
    for j in range(c):
        print("block", j)
        s, e = j*Nc, Nc*(j+1)
        if e>n: e=n
        # print(j, s, e, X[:, s:e].shape)
        U, S, V = svd(X[:, s:e], full_matrices=False)
        l_S[j], l_U[j], V = truncate_basis(S, U, V, eps)
        # plotS(S, .999)
        # TODO: truncate!
    return merge_blocks(l_U, l_S)

def merge_blocks(l_U, l_S):
    print("merge blocks")
    levels = int(np.log2(len(l_U)))
    for j in range(levels):
        print("level: ", j, "(", len(l_U), "slices)")
        Ni = len(l_U)
        l_Ut, l_St = l_U, l_S
        Ni2 = (len(l_U)+1) // 2
        l_U, l_S = [None for i in range(Ni2)], [None for i in range(Ni2)]
        for i in range(0, Ni, 2):
            if i+1 >= Ni:
                print(i, "nothing to merge")
                U, S = l_Ut[i], l_St[i]
            else:
                tic = timeit.default_timer()
                U, S = merge_horizontally(l_Ut[i], l_St[i], l_Ut[i+1], l_St[i+1])
                toc = timeit.default_timer()
                print("merging block", i, "and", i+1, "took ", toc-tic, "seconds")
            l_U[i], l_S[i] = U, S
        print(len(l_U), "blocks remaining")
    print()
    return U, S

def merge_horizontally(U1, S1, V1, U2, S2, V2):
    # V may be none
    # merging two bocks, [X1, X2] = [X]
    # print("1D?:", S1.shape, S2.shape)
    # k, l = S1.shape[0], S2.shape[0]
    m, k = U1.shape
    m, l = U2.shape
    # print("m, k = ", m, k)
    # print("m, l = ", m, l)

    # k, n1 = V1.shape
    # l, n2 = V2.shape
    # print("k, n1 = ", k, n1)
    # print("k, n2 = ", l, n2)
    # print("1D?: k, l = ", S1.shape, S2.shape)

    E = np.zeros((k+l, k+l))  # n, n
    U1TU2 = dot(U1.T, U2)  # k, l
    U0 = U2 - dot(U1, U1TU2)  # m, l
    Q, R = qr(U0)
    # print(Q.shape, R.shape, np.allclose(np.dot(Q, R), U0))

    E[:k, :k] = np.diag(S1)
    E[:k, k:] = U1TU2*S2.reshape(1, l)
    E[k:, k:] = R*S2.reshape(1, l)
    UE, S, VE = svd(E, full_matrices=False)
    # print(UE.shape, SE.shape, VE.shape)
    U1Q = np.empty((m, k+l))
    U1Q[:, :k] = U1
    U1Q[:, k:] = Q
    U = dot(U1Q, UE)
    if isinstance(U1, np.ndarray):
        k, n1 = V1.T.shape
        l, n2 = V2.T.shape
        # print("k, n1 = ", k, n1)
        # print("k, n2 = ", l, n2)
        V1V2 = np.empty((k+l, n1+n2))
        V1V2[:k, :n1] = V1.T
        V1V2[k:, n1:] = V2.T
        V = np.dot(VE.T, V1V2)
        return U, S, V.T
    else:
        return U, S

def merge_horizontally2(U1, S1, V1, U2, S2, V2):
    # V may be none
    # eq 9 and eq 10
    # merging two bocks, [X1, X2] = [X]
    # print("1D?:", S1.shape, S2.shape)
    # k, l = S1.shape[0], S2.shape[0]
    m, k = U1.shape
    m, l = U2.shape
    # print("m, k = ", m, k)
    # print("m, l = ", m, l)
    # print("1D?: k, l = ", S1.shape, S2.shape)

    E = np.zeros((m, k+l))  # n, n
    E[:, :k] = U1*S1.reshape(1, k)
    E[:, k:] = U2*S2.reshape(1, l)
    U, S, VE = svd(E, full_matrices=False)
    if isinstance(U1, np.ndarray):
        k, n1 = V1.T.shape
        l, n2 = V2.T.shape
        # print("k, n1 = ", k, n1)
        # print("k, n2 = ", l, n2)
        V1V2 = np.empty((k+l, n1+n2))
        V1V2[:k, :n1] = V1.T
        V1V2[k:, n1:] = V2.T
        V = np.dot(VE.T, V1V2)
        return U, S, V.T
    else:
        return U, S

def merge_vertically(U1, S1, V1, U2, S2, V2):
    # U may be none
    # eq 9 and eq 10
    # merging two bocks, [X1, X2] = [X]
    # print("1D?:", S1.shape, S2.shape)
    # k, l = S1.shape[0], S2.shape[0]
    k, n = V1.T.shape
    l, n = V2.T.shape
    # print("k, n = ", k, n)
    # print("l, n = ", l, n)

    # k, n1 = V1.shape
    # l, n2 = V2.shape
    # print("k, n1 = ", k, n1)
    # print("k, n2 = ", l, n2)
    # print("1D?: k, l = ", S1.shape, S2.shape)

    E = np.zeros((k+l, n))  # n, n
    E[:k, :] = V1.T*S1.reshape(k, 1)
    E[k:, :] = V2.T*S2.reshape(l, 1)
    UE, S, V = svd(E, full_matrices=False)
    if isinstance(U1, np.ndarray):
        m1, k = U1.shape
        m2, l = U2.shape
        # print("m1, k = ", m1, k)
        # print("m2, l = ", m2, l)
        U1U2 = np.empty((m1+m2, k+l))
        U1U2[:m1, :k] = U1
        U1U2[m1:, k:] = U2
        U = np.dot(U1U2, UE)
        return U, S, V
    else:
        return S, V


class Data:
    """
    data handling class. takes care of
        - normalising / scaling
        - splitting into train and test data
        -
    """
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
        return

    def normalise(self):
        # TODO: maybe rather mean center?
        # return X, 0.0, 1.0
        X_min = self.X.min(axis=1)[:, None]  # n
        X_max = self.X.max(axis=1)[:, None]  # n
        X_range = X_max - X_min
        X_range[X_range < 1e-6] = 1e-6
        self.X_min = X_min
        self.X_range = X_range
        self.X_n = self.scale_down(self.X)
        return self.X_n

    def scale_down(self, Snapshots):
        return (Snapshots-self.X_min)/self.X_range

    def scale_up(self, Snapshots_n):
        return Snapshots_n * self.X_range + self.X_min

    def split_off(self, set_i):
        # [[s1, s2], [d1, d2]] = self.dims
        self.X.shape = (self.s1, self.s2, self.d1, self.d2)
        self.xi.shape = (self.d1, self.d2, 2)

        i_train = np.delete(np.arange(self.d2), set_i)
        self.X_train = self.X[..., i_train].copy()
        self.X_valid = self.X[..., set_i].copy()
        self.xi_train = self.xi[:, i_train, :].copy()
        self.xi_valid = self.xi[:, set_i, :].copy()

        n_p = 1
        if isinstance(self.phase_length, np.ndarray):
            offset = np.c_[phase_length[i_train],
                           np.zeros_like(phase_length[i_train])]
            self.xi_train = np.concatenate((self.xi_train-offset,
                                            self.xi_train,
                                            self.xi_train+offset), axis=0)
            # X_train = np.concatenate((X_train, X_train, X_train), axis=2)
            n_p = 3  # number of repetitions of each periods
        self.X.shape = (self.s1*self.s2, self.d1*self.d2)
        self.xi.shape = (self.d1*self.d2, 2)
        self.X_train.shape = (self.s1*self.s2, self.d1*(self.d2-1))
        self.xi_train.shape = (n_p*self.d1*(self.d2-1), 2)
        self.X_valid.shape = (self.s1*self.s2, self.d1*1)
        self.xi_valid.shape = (self.d1*1, 2)
        return self.X_train, self.X_valid, self.xi_train, self.xi_valid


class POD(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def svd(self, X):
        S, U, V = tf.linalg.svd(X, full_matrices=False)
        return U.numpy(), S.numpy(), V.numpy()

    def hierarchical_pod(self, eps):
        """
        a tree based merge-and-truncate algorithm to obtain an approximate
        truncated SVD of the matrix.

        Parameters
        ----------
        eps : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Hierarchical Singular Value Decomposition
        return

    def higher_order_svd(a, full_matrices=True):
        # a = np.random.rand(2, 3, 4, 5)
        # U, S = higher_order_svd(a)
        # print(S.shape)
        # print([i.shape for i in U])
        # a1 = S
        # for i, _ in enumerate(U):
            # a1 = np.tensordot(a1, U[i], (0, 1))
        # print(np.allclose(a, a1))
        # core_tensor = a
        left_singular_basis = []
        for k in range(a.ndim):
            unfold = np.reshape(np.moveaxis(a, k, 0), (a.shape[k], -1))
            U, _, _ = np.linalg.svd(unfold, full_matrices=full_matrices)
            left_singular_basis.append(U)
            U_c = U.T.conj()
            a = np.tensordot(a, U_c, (0, 1))
        return left_singular_basis, a

    def _2_way_pod(self, eps1, eps2):
        # #[eps, error, n basis vectors (avg, per dataset)],
        # res = np.array([
        # [1.000000000, 0.000649143, 416.3], [1, 0.000649143, 416.3],
        # [0.994897959, 0.001054028, 251.5], [0.989795918, 0.00118231, 211],
        # [0.984693878, 0.001342111, 181.3], [0.979591837, 0.001533657, 157.4],
        # [0.974489796, 0.001778205, 137.4], [0.973684211, 0.001824512, 134.5],
        # [0.969387755, 0.002076038, 120.3], [0.964285714, 0.002402765, 105.2],
        # [0.959183673, 0.002712533, 91.7], [0.954081633, 0.003031809, 79.9],
        # [0.948979592, 0.003378348, 69.5], [0.947368421, 0.00349487, 66.3],
        # [0.943877551, 0.003744284, 60.2], [0.93877551, 0.004137558, 52],
        # [0.933673469, 0.004531132, 45.2], [0.928571429, 0.004951166, 39.3],
        # [0.923469388, 0.005380647, 34.4], [0.921052632, 0.005613473, 32.2],
        # [0.918367347, 0.005870244, 30.1], [0.913265306, 0.006364088, 26.4],
        # [0.908163265, 0.006875367, 23.5], [0.903061224, 0.007418286, 20.9],
        # [0.897959184, 0.008020077, 18.6], [0.894736842, 0.00837866, 17.4],
        # [0.892857143, 0.008667827, 16.6], [0.887755102, 0.009338536, 15],
        # [0.882653061, 0.010072033, 13.6], [0.87755102, 0.010745011, 12.4],
        # [0.872448980, 0.011194270, 11.6], [0.868421053, 0.01183017, 10.8],
        # [0.867346939, 0.012081178, 10.5], [0.862244898, 0.012796338, 9.8],
        # [0.857142857, 0.013526181, 9.2], [0.852040816, 0.014283143, 8.5],
        # [0.846938776, 0.015037375, 8.0], [0.842105263, 0.015727348, 7.7],
        # [0.841836735, 0.015727348, 7.7], [0.836734694, 0.016882615, 7.1],
        # [0.831632653, 0.017513710, 6.7], [0.826530612, 0.018314773, 6.4],
        # [0.821428571, 0.018648107, 6.2], [0.815789474, 0.020361032, 5.7],
        # [0.816326531, 0.020361032, 5.7], [0.81122449, 0.02121174, 5.5],
        # [0.806122449, 0.022170866, 5.3], [0.801020408, 0.022990563, 5.2],
        # [0.795918367, 0.023983465, 5.0], [0.789473684, 0.027309363, 4.4],
        # [0.790816327, 0.027309363, 4.4], [0.785714286, 0.027996551, 4.3],
        # [0.780612245, 0.029564418, 4.2], [0.775510204, 0.031077579, 4.1],
        # [0.765306122, 0.032118860, 4.0], [0.770408163, 0.03211886, 4],
        # [0.763157895, 0.032865164, 3.7], [0.760204082, 0.033701762, 3.4],
        # [0.755102041, 0.034560932, 3.3], [0.75, 0.035673262, 3.2],
        # [0.736842105, 0.038180729, 3.0], [0.710526316, 0.042435443, 2.3],
        # [0.684210526, 0.062167254, 2.0], [0.657894737, 0.064424002, 1.9],
        # [0.578947368, 0.135025099, 1.0], [0.605263158, 0.135025099, 1],
        # [0.631578947, 0.135025099, 1.0], [0.552631579, 0.137783313, 0.9],
        # [0.526315789, 0.139093005, 0.8], [0.5, 0.15059241, 0.5]])
        self.X_n.shape = (self.s1*self.s2, self.d1, self.d2)
        U_hats = np.zeros((self.s1*self.s2, self.d1*self.d2))
        s, e = 0, 0
        n = 2
        for i in range(self.d2):
            X = self.X_n[:, ::n, i]
            f_name = "_{:03.0f}(every {:.0f} th SS).npy".format(i, n)
            try:
                U = np.load("U"+f_name)
                S = np.load("S"+f_name)
                V = np.load("V"+f_name)
                # print("loaded: "+f_name)
            except:
                U, S, V = self.svd(X)
                np.save("U"+f_name, U)
                np.save("S"+f_name, S)
                np.save("V"+f_name, V)
                # print("saved: "+f_name)
            S_hat, U_hat, V_hat = self.truncate_basis(S, U, V, eps1)
            e = s + U_hat.shape[1]
            U_hats[:, s:e] = U_hat
            # print(i, X.shape, U.shape, U_hat.shape)
            s = e
        U_hats = U_hats[:, :e]
        self.X_n.shape = (self.s1*self.s2, self.d1*self.d2)
        self.U_hats = U_hats

        # print(U_hats.shape)
        # self = my_POD
        # U_hats = self.U_hats
        # U_hat = self.U_hat
        # S, U, V = self.S, self.U, self.V
        # S_hat, U_hat, V_hat = self.S_hat, self.U_hat, self.V_hat
        # print(U_hats.shape)

        U, S, V = self.svd(U_hats)
        S_hat, U_hat, V_hat = self.truncate_basis(S, U, V, eps2)
        # print(S.shape, S_hat.shape)
        # print(U.shape, U_hat.shape)
        # print(V.shape, V_hat.shape)

        X_n_approx = np.dot(U_hat, np.dot(U_hat.T, self.X_n))
        error = np.std(self.X_n-X_n_approx)

        # self.S, self.U, self.V = S, U, V
        # self.S_hat, self.U_hat, self.V_hat = S_hat, U_hat, V_hat
        # self.X_n_approx = X_n_approx

        # print(error)
        # print(np.mean(np.abs(self.X_n-X_n_approx)))
        # print(np.mean(np.abs(self.X_n/X_n_approx)))
        # self.X_n.shape = (self.s1*self.s2, self.d1, self.d2)
        # X_n_approx.shape = (self.s1*self.s2, self.d1, self.d2)
        # for i in range(60):
        #     plot_snapshot_cav(X_n_approx[:, 500, i], x, y, tri)
        #     plt.show()
        #     plot_snapshot_cav(self.X_n[:, 500, i], x, y, tri)
        #     plt.show()

        # rel_energy = np.cumsum(S) / np.sum(S)
        # print(rel_energy.shape)
        # fig, ax = plt.subplots()
        # plt.plot(rel_energy, "b.")
        # plt.show()
        return error

    def truncate_basis(self, S, U, V, eps):
        """
        reduce rank of basis to low rank r
        """
        cum_en = np.cumsum(S)/np.sum(S)
        r = np.sum(cum_en<eps)
        U_hat = U[:, :r]  # (n, r)
        S_hat = S[:r]  # (r, r) / (r,) wo  non zero elements
        V_hat = V[:, :r]  # (d1, r)

        self.cum_en = cum_en
        # n = len(S)
        # fig, axs = plt.subplots(2, 1, sharex=True,
        #                         figsize=(plot_width/2.54, 10/2.54))
        # # for i in range(3):
        # axs[0].plot(np.arange(n), S, "r.")
        # axs[0].plot(np.arange(r), S[:r], "g.")
        # axs[1].plot(np.arange(n), cum_en, "r.")
        # axs[1].plot(np.arange(r), cum_en[:r], "g.")
        # axs[0].set_yscale('log')
        # # axs[0].legend()
        # # axs[1].legend()
        # axs[0].set_title("First n singular values S")
        # axs[1].set_title("Cumulative energy [%]")
        # axs[0].set_xlim(0, n)
        # # axs[0].set_ylim(1, 1000)
        # # axs[0].set_ylim(0, S[1])
        # # axs[0].set_ylim(bottom=0)
        # axs[1].set_xlabel("Snapshot number")
        # axs[0].set_ylabel("Singular value")
        # axs[1].set_ylabel("Energy in %")
        # # axs[1].set_ylim([0, 100])
        # plt.tight_layout()
        # plt.show()
        return S_hat, U_hat, V_hat

    def to_reduced_space(self):
        return

    def from_reduced_space(self):
        return

    # predict
    def predict(S, U, V, r):
        S_hat = S.numpy()[:r]  # (r, r) / (r,) wo  non zero elements
        U_hat = U.numpy()[:, :r]  # (n, r)
        V_hat = V[:, :r]  # (d1, r)
        X_approx = np.dot(U_hat*S_hat, V_hat.T)  # n, d1
        return X_approx


if __name__ == "__main__":
    path = "C:/Users/florianma/Documents/data/freezing_cavity/"
    # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
    X = np.random.rand(1000, 1000)
    U_, S_, V_ = svd(X, False)


    m1, n1 = 345, 456
    print("split vertically: X = [[X1], [X2]]")
    U1, S1, V1 = svd(X[:m1, :], False)
    U2, S2, V2 = svd(X[m1:, :], False)
    print(U1.shape, S1.shape, V1.T.shape)
    print(U2.shape, S2.shape, V2.T.shape)

    U, S, V = merge_vertically(U1, S1, V1, U2, S2, V2)
    print(U.shape, U_.shape, np.allclose(np.abs(U/U_), 1))
    print(S.shape, S_.shape, np.allclose(S, S_, atol=1e-5, rtol=1e-5))
    print(V.shape, V_.shape, np.allclose(np.abs(V/V_), 1))

    print("split horizontally: X = [X1, X2]")
    U1, S1, V1 = svd(X[:, :n1], False)
    U2, S2, V2 = svd(X[:, n1:], False)
    print(U1.shape, S1.shape, V1.T.shape)
    print(U2.shape, S2.shape, V2.T.shape)

    U, S, V = merge_horizontally(U1, S1, V1, U2, S2, V2)
    print(U.shape, U_.shape, np.allclose(np.abs(U/U_), 1))
    print(S.shape, S_.shape, np.allclose(S, S_, atol=1e-5, rtol=1e-5))
    print(V.shape, V_.shape, np.allclose(np.abs(V/V_), 1))
    U, S, V = merge_horizontally2(U1, S1, V1, U2, S2, V2)
    print(U.shape, U_.shape, np.allclose(np.abs(U/U_), 1))
    print(S.shape, S_.shape, np.allclose(S, S_, atol=1e-5, rtol=1e-5))
    print(V.shape, V_.shape, np.allclose(np.abs(V/V_), 1))




    # X[:500, :400] = 0; X  # 1, 1
    # X[500:, :300] = 0; X  # 2, 1
    # X[:500, -200:] = 0; X  # 1, 2
    # X[500:, -100:] = 0; X  # 2, 2
    # d, c = 2, 2
    # tic = timeit.default_timer()
    # U, S, V = hierarchical_svd(X, d, c, .9999)
    # toc = timeit.default_timer()
    # print("hierarchical svd X (shape ", X.shape, ") took {:.1f} seconds.".format(toc-tic))
    # U, S, V = svd(X, full_matrices=False)
    # np.linalg.svd()
    # X = np.random.rand(2000, 2000)
    # for n in range(100, 2000, 100):
    #     dot(X[:, :n], X[:n, :])
    #     dot(X[:n, :], X[:, :n])
    #     dot(X[:n, :n], X[:n, :n])
    # # for n in range(100, 1000, 25):
    # for n in range(100, 2000, 100):
    #     svd(X[:n, :], False)
    #     svd(X[:, :n], False)
    #     svd(X[:n, :n], False)





    # # S, U, V = tf.linalg.svd(X_all, full_matrices=False)
    # my_POD = POD(X_all, _xi_all_, x, y, tri, dims_all)
    # for eps in np.linspace(0.8, 1.0, 21):
    #     if eps == 1.0:
    #         eps = .9925
    #     error = my_POD._2_way_pod(eps, 1.0)
    #     print(eps, error, my_POD.U_hats.shape[1])
