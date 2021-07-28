# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:41:12 2021

@author: florianma
https://towardsdatascience.com/reduced-order-modeling-using-tensorflow-part-1-5697c49fb4d4
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import timeit
from scipy.optimize import curve_fit

plot_width = 16.0  # cm
# plt.style.use(['science', 'ieee'])
plt.rcParams['figure.constrained_layout.use'] = True

# path = "C:/Users/florianma/Documents/Repositoties/NIROM/mu(0.0008)/"
path = '/home/florianma@ad.ife.no/Documents/NIROM/mu(0.0008)/'
path = '/home/florianma@ad.ife.no/Documents/NIROM/doc/mu(0.0050)/'
path = '/home/florianma@ad.ife.no/Documents/NIROM/mu.003/'
path = '/home/florianma@ad.ife.no/Documents/NIROM/doc/mu(0.0017)/'

dt = 7.479453076628512e-05
n_snapshots = 200  # to be used
J = 10000  # number of snapshots per saved file
n = 90000  # start timestamp in saved file
# select random snapshots
indices = np.sort(np.random.randint(0, J, size=n_snapshots))
t = (indices+n)*dt


def plot_up(u, v, p, x, y, tri, umin=None, umax=None, pmin=None, pmax=None):
    umin = u.min() if (not umin) else umin
    umax = u.max() if (not umax) else umax
    pmin = p.min() if (not pmin) else pmin
    pmax = p.max() if (not pmax) else pmax
    magnitude = (u**2 + v**2)**.5
    cmap = mpl.cm.inferno

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                   figsize=(plot_width/2.54, 8/2.54))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("velocity")
    ax2.set_title("pressure")

    c_ticks = np.linspace(umin, umax, num=5, endpoint=True)
    norm = mpl.colors.Normalize(vmin=umin, vmax=umax)
    cp1 = ax1.quiver(x, y, u, v, cmap=cmap, color=cmap(norm(magnitude)))
    cbar1 = plt.colorbar(cp1, ax=ax1, ticks=norm(c_ticks))
    cbar1.ax.set_yticklabels(["{:.2f}".format(i) for i in c_ticks])
    cbar1.set_label('velocity')

    # c_ticks = np.linspace(np.floor(pmin*100)/100, np.ceil(pmax*100)/100, num=7,
    #                       endpoint=True)
    # norm = mpl.colors.Normalize(vmin=pmin, vmax=pmax)
    lvls = np.linspace(pmin, pmax, num=40, endpoint=True)
    cp2 = ax2.tricontourf(x, y, tri, p, levels=lvls, cmap=cmap)
    cbar2 = plt.colorbar(cp2, ax=ax2)
    cbar2.set_label('pressure')
    return fig, (ax1, ax2)


# # load snapshots '/home/florianma@ad.ife.no/Documents/NIROM/mu(0.0008)/'
# u = np.load(path+"{:06.0f}to{:06.0f}_u.npy".format(n, n+J)).T  # n, t
# v = np.load(path+"{:06.0f}to{:06.0f}_v.npy".format(n, n+J)).T  # n, t
# p = np.load(path+"{:06.0f}to{:06.0f}_p.npy".format(n, n+J)).T  # n, t
# x = np.load(path+"x.npy")
# y = np.load(path+"y.npy")
# tri = np.load(path+"tri.npy")
# load snapshots '/home/florianma@ad.ife.no/Documents/NIROM/doc/mu(0.0050)/'
u = np.load(path+"{:06.0f}u.npy".format(n, n+J)).T  # n, t
v = np.load(path+"{:06.0f}v.npy".format(n, n+J)).T  # n, t
p = np.load(path+"{:06.0f}p.npy".format(n, n+J)).T  # n, t
x = np.load(path+"__x.npy")
y = np.load(path+"__y.npy")
tri = np.load(path+"__tri.npy")

S = 3*[None]
U = 3*[None]
V = 3*[None]
for i, tensor in enumerate([u, v, p]):
    tic = timeit.default_timer()
    tensor = tensor[:, indices]
    mean_tensor = tf.reduce_mean(tensor, axis=1, keepdims=True)
    mean_centered_data = tf.subtract(tensor, mean_tensor)
    S[i], U[i], V[i] = tf.linalg.svd(mean_centered_data, full_matrices=False)
    # Full: S.shape = (T,), u.shape = (n, n), v.shape = (t, t) / comp. time: 6 - 10 s for T=150 / comp. time: 23 - 30 s for T=1500
    # Economy: S.shape = (T,), u.shape = (n, t), v.shape = (t, t) / comp. time: .4 - 1 s for T=150
    # _u2_, _s2_, _v2_ = np.linalg.svd(mean_centered_data, full_matrices=True)
    toc = timeit.default_timer()
    print(S[i].shape, U[i].shape, V[i].shape, n_snapshots, toc-tic)


fig, ax = plot_up(np.mean(u, axis=1), np.mean(v, axis=1), np.mean(p, axis=1),
                  x, y, tri, umin=0, umax=2.2, pmin=-.3, pmax=3.1)
plt.show()

# eigenfaces:
for i in range(6):
    u_eig = U[0][:, i].numpy()
    v_eig = U[1][:, i].numpy()
    p_eig = U[2][:, i].numpy()
    fig, ax = plot_up(u_eig, v_eig, p_eig, x, y, tri)
    ttl = ("eigen vector #{:.0f}".format(i))
    plt.suptitle(ttl)
    plt.show()

lbl = ["velocity in x direction", "velocity in y direction", "pressure"]
k = 25
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(plot_width/2.54, 10/2.54))
for i in range(3):
    axs[0].plot(np.arange(k), S[i][:k], ".", label=lbl[i])
    cum_en = np.cumsum(S[i])[:k]/np.sum(S[i])*100
    axs[1].plot(np.arange(k), cum_en, ".", label=lbl[i])
axs[0].legend()
axs[1].legend()
axs[0].set_title("First n singular values")
axs[1].set_title("Cumulative energy [%]")
axs[0].set_xlim(0, k)
axs[1].set_xlabel("Snapshot number")
axs[0].set_ylabel("Singular value")
axs[1].set_ylabel("Energy in %")
axs[0].set_ylim(bottom=0)
axs[1].set_ylim([0, 100])
plt.show()


# colors = ["r", "g", "b"]
fig, ax = plt.subplots()
# for j in range(1):
for i in range(3):
    ax.plot(t, V[i][:, 0], linestyle='-', marker='.', label=lbl[i])
plt.title("Amplitude of the first mode over time")
ax.set_xlabel("Snapshot number")
ax.set_ylabel("Modes amplitude")
ax.legend()
plt.show()


# find frequency.
# guesses
phase_length = .33
amplitude = (np.max(V[0][:, 0])-np.min(V[0][:, 0]))/2
offset = (np.max(V[0][:, 0])+np.min(V[0][:, 0]))/2
phase = t[np.argmax(V[0][:, 0])]

p0 = [phase_length, amplitude, phase, offset]


def my_cos(x, T, amplitude, phase, offset):
    # print(offset)
    return np.cos((x-phase)*2*np.pi / T) * amplitude + offset


fit = curve_fit(my_cos, t, V[0][:, 0], p0=p0)

# recreate the fitted curve using the optimized parameters
data_fit = my_cos(t, *fit[0])

fig, ax = plt.subplots()
plt.plot(t, V[0][:, 0], "ko", label="data")
plt.plot(t, my_cos(t, *p0), "b.", label='first guess')
plt.plot(t, my_cos(t, *fit[0]), "g.", label='after fitting')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Modes amplitude")
plt.suptitle("phase length: {:.4f} s".format(fit[0][0]))
plt.show()


if False:
    mu = .0016666666666666668
    rho = 1.0
    Re = rho*1.0*0.1/mu

    # indices_all = np.arange(0, J)
    time = (np.arange(0, J)+n)*dt
    time_span = time[-1]-time[0]
    N = len(time)
    M = int(np.ceil(fit[0][0] / time_span * N))
    # M = N
    my_data = {
        "x": x,
        "y": y,
        "time": time[:M],
        "tri": tri,
        "u": u[:, :M],
        "v": v[:, :M],
        "p": p[:, :M],
        "Re": Re,
        "mu": mu,
        "rho": rho,
        }

    path = "/home/florianma@ad.ife.no/Documents/NIROM/"
    with open(path + "Re060.pickle", "wb") as handle:
        pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Re 200: 0.3095188656003769
    # Re 150: 0.31802089914746967
    # Re 125: 0.32157462104473244
    # Re 100: 0.33064783989089486
    # Re 75: 0.3432243056825879
    # Re 50: 0.3717622916635874
    # Re 33: 0.3876948108794104








































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