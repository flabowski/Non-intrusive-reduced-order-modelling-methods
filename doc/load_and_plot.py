# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:41:12 2021

@author: florianma
https://towardsdatascience.com/reduced-order-modeling-using-tensorflow-part-1-5697c49fb4d4
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import timeit

path = "C:/Users/florianma/Documents/Repositoties/Cylinder_NIROM/"
N = 106959
J = 10000
n_timesteps = 1500


def plot_up(u, v, p, x, y, tri):
    magnitude = (u**2 + v**2)**.5

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                   figsize=(12, 6))
    ax1.quiver(x, y, u, v, magnitude)
    ax2.tricontourf(x, y, tri, p, levels=40)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("velocity")
    ax2.set_title("pressure")
    return fig, (ax1, ax2)


# load snapshots
n = 100000
# u = np.load(path+"{:06.0f}to{:06.0f}_u.npy".format(n-J, n))
# v = np.load(path+"{:06.0f}to{:06.0f}_v.npy".format(n-J, n))
# p = np.load(path+"{:06.0f}to{:06.0f}_p.npy".format(n-J, n))
u = np.load(path+"u_{:06.0f}to{:06.0f}.npy".format(n-J, n)).T  # n, t
v = np.load(path+"v_{:06.0f}to{:06.0f}.npy".format(n-J, n)).T  #
p = np.load(path+"p_{:06.0f}to{:06.0f}.npy".format(n-J, n)).T
x = np.load(path+"x.npy")
y = np.load(path+"y.npy")
tri = np.load(path+"tri.npy")

indices = np.random.randint(0, len(u[0]), size=n_timesteps)

# u_t = np.concatenate((u[..., None], v[..., None]), axis=2)  # t, n, 2
u_mean = np.mean(u, axis=1)  # n
v_mean = np.mean(v, axis=1)  # n
p_mean = np.mean(p, axis=1)  # n
u_centered = u - u_mean[:, None]
v_centered = v - v_mean[:, None]
p_centered = p - p_mean[:, None]
fig, ax = plot_up(u_mean, v_mean, p_mean, x, y, tri)
plt.show()

S = 3*[None]
U = 3*[None]
V = 3*[None]
for i, tensor in enumerate([u, v, p]):
    tic = timeit.default_timer()
    tensor = tensor[:, indices]
    mean_tensor = tf.reduce_mean(tensor, axis=1, keepdims=True)
    mean_centered_data = tf.subtract(tensor, mean_tensor)
    _s_, _u_, _v_ = tf.linalg.svd(mean_centered_data, full_matrices=False)
    # Full: S.shape = (T,), u.shape = (n, n), v.shape = (t, t) / comp. time: 6 - 10 s for T=150
    # Economy: S.shape = (T,), u.shape = (n, t), v.shape = (t, t) / comp. time: .4 - 1 s for T=150
    # _u2_, _s2_, _v2_ = np.linalg.svd(mean_centered_data, full_matrices=True)
    toc = timeit.default_timer()
    S[i], U[i], V[i] = _s_, _u_, _v_

    print(S[i].shape, U[i].shape, V[i].shape, n_timesteps, toc-tic)
    print(_s_.shape, _u_.shape, _v_.shape, n_timesteps, toc-tic)

# eigenfaces:
for i in range(6):
    u_eig = U[0][:, i]
    v_eig = U[1][:, i]
    p_eig = U[2][:, i]
    print(u_eig.shape, v_eig.shape, p_eig.shape)
    fig, ax = plot_up(u_eig, v_eig, p_eig, x, y, tri)
    plt.suptitle("eigen vector #{:.0f}, singular values (u, v, p): {:.3f}, {:.3f}, {:.3f}".format(i, S[0][i], S[1][i], S[2][i]))
    plt.show()

fig, ax = plt.subplots()
for i in range(3):
    plt.plot(np.arange(n_timesteps), S[i], "o")
plt.title("singular values")
plt.show()


# E = sess.run(tf.reduce_sum(S))
# s_energy = sess.run(tf.div(S, 1E)*100)


# for i in range(25):
#     i = np.random.randint(0, len(x))
#     plt.plot(np.arange(0, len(u)), u[:, i], ".")
# plt.show()



   # data_tensor, axis=1, keep_dims=True)


# for n in range(N):
#     # u[j] = np.load(path+"{:06.0f}u.npy".format(n+1))
#     # v[j] = np.load(path+"{:06.0f}v.npy".format(n+1))
#     # p[j] = np.load(path+"{:06.0f}p.npy".format(n+1))
#     j += 1
#     if ((n % J) < 1e-4) and (n != 0):
#         print(n, end=" ")


# for filename in os.listdir():
#     if filename.endswith(".npy"):
#           # print(os.path.join(directory, filename))
#         continue
#     else:
#         continue