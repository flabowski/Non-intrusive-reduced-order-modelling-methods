#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:04:11 2021

@author: florianma
"""
import numpy as np
from numpy import sin, pi
from  scipy.optimize import curve_fit
from scipy.special import jv
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from ROM.snapshot_manager import Data
from low_rank_model_construction.basis_function_interpolation import interpolateV, RightSingularValueInterpolator
from low_rank_model_construction.proper_orthogonal_decomposition import truncate_basis
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close("all")
cmap = matplotlib.cm.get_cmap('jet')
# cmap = matplotlib.cm.get_cmap('parula')
cmap = matplotlib.cm.get_cmap('viridis')
plot_width = 12

# dx = .0001
n = 401
n2 = 100+1

L = 15
T = 10

x = np.linspace(-L, L, 2001)
times = np.linspace(-T/2, T/2, n)

# x = np.arange(-L, L, dx)
dx = x[1]-x[0]
m = len(x)

X = np.empty((m, n))
Xe_fine = np.empty((m, n2))
xi = np.linspace(0, 1, n2).reshape(-1, 1)
# times[-1] = 1

def front1D(x, time):
    print(time)
    # c = .1# /2
    y = np.zeros_like(x)
    y[(x<=time)] = 1.0
    return y

def slope1D(x, time):
    print(time)
    # c = .1# /2
    y = -10*(x-time)+1
    y[y>1] = 1
    y[y<0] = 0
    # y[(x<=time)] = 1.0
    return y

def sigmoid1D(x, time):
    y = 1.0 / (1+np.e**((x-time)*50))
    return y

def sigmoid21D(x, time):
    y = 2.0 / (1+np.e**((x-time)*50))
    y[y>1] = 1
    return y

def plug_1D(x, time):
    print(time)
    c = .1 / 2 
    y = np.zeros_like(x)
    y[((time-c)<x) & (x<=(time+c))] = 1.0
    return y

def random_1D(x, time):
    print(time)
    c = .1# /2
    y = np.zeros_like(x)
    wave = [0.51538182, 0.36639445, 0.8030119 , 0.40366875, 0.23448425,
           0.380502  , 0.01432753, 0.53730148, 0.46815127, 0.99682601,
           0.47923601, 0.68289249, 0.28251217, 0.1522982 , 0.50534002,
           0.00396807, 0.81103336, 0.17195926, 0.19020709, 0.13527934]
    s = len(y[((time-c)<x) & (x<(time+c))])
    if time<.5:
        y[((time-c)<x) & (x<(time+c))] = wave[20-s:]
    else:
        y[((time-c)<x) & (x<(time+c))] = wave[:s]
    return y

def gauss_brunton(x, time):
    c = 3
    y = np.exp(-(x+15-c*time)**2)
    return y

def gauss_1D(x, time):
    a = 1 #/2
    speed = 3
    c = 1 / 2.**.5
    y = a*np.e**(-(x+speed*time)**2/(2*c**2))
    return y


my_func = gauss_1D
# fig, ax = plt.subplots()
# ax.plot(x, plug_1D(x, .5))
# ax.plot(x, gauss1D(x, .5))
# plt.show()


for j, time in enumerate(times):
    X[:, j] = my_func(x, time)
for j, time in enumerate(xi[:, 0]):
    Xe_fine[:, j] = my_func(x, time)

grid = [times]
my_data = Data(X, grid)
my_data.decompose(eps=1.0)#-1e-6)
# X_ = my_data.from_reduced_space(my_data.to_reduced_space(X))
# # error = my_data.std_rb(X)
U = my_data.U
S = my_data.S
VT = my_data.VT

l = VT[:, 0]<0
U[:, l] *= -1
VT[l, :] *= -1

Uhat, Shat, VThat = truncate_basis(U, S, VT, 1-1e-6)
# print(my_data.S)
r = len(S)

intpld = RightSingularValueInterpolator(grid, VT, method="cubic")
VT_approx_fine = intpld(xi)
X_approx_fine = my_data.from_reduced_space(VT_approx_fine)
VTe_fine = my_data.to_reduced_space(Xe_fine)
Xe_proj_fine = my_data.from_reduced_space(VTe_fine)
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

# plot X: solutions in separate plots
fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
ax.plot(x, X[:, 0], "k-")
plt.xlabel("x")
plt.ylabel("u(x)")
ax.annotate("time",
            xy=(0.5, .8), xycoords='data',
            xytext=(0.15, 0.8), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            va="center"
            )
# plt.legend()
plt.show()

# plot X: solutions in separate plots
fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
for j, time in enumerate(times):
    # print(np.isclose(time, np.linspace(0, 1, 11)).any(), j, time)
    if np.isclose(time, np.linspace(0, 1, 11)).any():
        ax.plot(x, X[:, j], color=cmap(j/len(times)), marker=".", label="FEM solution")
plt.xlabel("x")
plt.ylabel("u(x)")
# plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
for j, time in enumerate(xi[:, 0]):
    # print(np.isclose(time, np.linspace(0, 1, 11)).any(), j, time)
    if np.isclose(time, np.linspace(0, 1, 11)).any():
        ax.plot(x, X_approx_fine[:, j], color=cmap(j/n2), marker=".", label="FEM solution")
plt.xlabel("x")
plt.ylabel("u(x)")
# plt.legend()
plt.show()


# plot S
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54))
ax1.plot(np.arange(0, r), S, color='#1f77b4', marker=".")
ax2.plot(np.arange(0, r), np.cumsum(S)/S.sum(), color='#1f77b4', marker=".")
# ax.plot(x[0], Xe[:, j], "g.", label="exact solution")
ax1.set_xlabel("rank")
ax1.set_ylabel("singular values")
ax2.set_xlabel("rank")
ax1.set_yscale('log')
ax2.set_xscale('log')
ax2.set_ylabel("Cumulative Energy [%]")
ax1.grid(which="both")
ax2.grid(which="both")
# plt.title("dacay of singular values")
ax1.set_ylim([1e-12, 1e4])
ax1.set_xlim([0, r])
plt.show()


   
def sinfunc(t, A, w, p, c=0):
    return A * np.sin(w*t + p) + c
# def sinfunc2(t, w, p):
#     A = 0.009744857
#     c = 0
#     return A * np.sin(w*t + p) + c

def fit_sin(j, tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    # tt = np.array(tt)
    # yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = 2.*np.pi*abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    if j==0:
        guess_freq = 0.09
    # print(guess_freq,  2.81*(j+1))
    # guess_freq = j*2.99283738791371+1
    guess_amp = yy.max() - yy.min()  #np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, guess_freq, 0., guess_offset])
    # guess = np.array([guess_freq, 0.])

    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    # w, p = popt
    # f = w/(2.*np.pi)
    return A, w, p, c
    # fitfunc = lambda t: A * np.sin(w*t + p) + c
    # return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}


reduced_rank = len(Shat)
# plot VT
f_svd = np.zeros(reduced_rank,)
f_svd2 = np.zeros(reduced_rank,)
p_svd = np.zeros(reduced_rank,)
U_fourier = np.zeros_like(Uhat)
U_fourier2 = np.zeros_like(Uhat)
xxx = np.linspace(-L, L, 10000)
fig, ax = plt.subplots(1, 1, figsize=(plot_width/2.54, plot_width/2.54))
for j in range(reduced_rank):
    # if j<100:
    a = (np.max(U[:, j]) - np.min(U[:, j]))/2
    l = (-15<x) & (x<15)
    A, w, p, c = fit_sin(j, x[l], U[l, j])
    # print(j, A, w, p, c, sep=", ")
    print(j, A, w, p, c, sep="\t")
    f_svd[j] = w
    p_svd[j] = p
    # A = 0.009744857
    U_fourier[:, j] = sinfunc(x, A, w, p, c)
    t = np.linspace(0, 1, len(x), endpoint=False)
    w = 2*np.pi *(j+1)/2
    U_fourier2[:, j] = A*np.sin(w*t)
    f_svd2[j] = w
    ax.plot(x, U[:, j], "k.", label="basis vector from SVD")
    ax.plot(xxx, sinfunc(xxx, A, w, p, c), "r.", label="sine function fit")
    if j < 50:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54))
        ax1.plot(x, U[:, j], "b.", label="basis vector from SVD")
        ax1.plot(x, U_fourier[:, j], "r.", label="sine function fit")
        ax1.plot(x, U_fourier2[:, j], "k.", label="fourier mode")
        ax2.plot(times, VT[j], "b.", label="snapshots", ms=10)
        # ax2.plot(xi[:, 0], VTe_fine[j], "g.", label="exact solution")
        # ax2.plot(xi[:, 0], VT_approx_fine[j], "r.", label="interpolated solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("U(x)")
        ax1.set_title("basis vector {:.0f}, energy = {:.1f} %".format(j, S[j]/np.sum(S)*100))
        ax2.set_xlabel("time")
        ax2.set_ylabel("VT(time)")
        ax2.set_title("right singular values \n dynamics of mode {:.0f}".format(j))
        ax1.legend()
        plt.savefig("C:/Users/florianma/OneDrive - Institutt for Energiteknikk\Pictures/tmp/frame_{:.0f}".format(10000+j))
        plt.close()
    U_fourier[:, j] /= (U_fourier[:, j]**2).sum()**.5
    U_fourier2[:, j] /= (U_fourier2[:, j]**2).sum()**.5
plt.show()

f_sorted = np.sort(f_svd)
df = np.diff(f_sorted)  # better than average, not so sensitive to outliers
a = np.median(df)
c = f_svd % a


fig, ax = plt.subplots(1, 1, figsize=(3*plot_width/2.54, plot_width/2.54))
plt.hist(df, 100)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(3*plot_width/2.54, plot_width/2.54))
plt.hist(c, 100)
plt.show()

fig, ax = plt.subplots()
ax.plot(c)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(plot_width/2.54, plot_width/2.54))
ax1.imshow(np.matmul(U[:, :reduced_rank].T, U[:, :reduced_rank]), vmin=0, vmax=.10)
ax2.imshow(np.matmul(U_fourier.T, U_fourier), vmin=0, vmax=.1)
# U_fourier2 = U_fourier.copy()
# U_fourier2[:200, :] = 0
# U_fourier2[-200:, :] = 0
ax3.imshow(np.matmul(U_fourier2.T, U_fourier2), vmin=0, vmax=.1)
plt.show()




fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = X_approx_fine.shape
h, w = (times.max() - times.min())/ncols, (x[-1]-x[0])/nrows
left, right, bottom, top = x[0]-h/2, x[-1]+h/2, times.min()-w/2, times.max()+w/2
im1 = ax1.imshow(Xe_fine.T, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im2 = ax2.imshow(Xe_proj_fine.T, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im3 = ax3.imshow(X_approx_fine.T, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))

# for i in range(len(times)):
#     color = X[:, i]
#     ax3.scatter(np.ones_like(x) * times[i], x, s=10, c=color, cmap=cmap,
#                 vmin=X.min(), vmax=X.max(), edgecolor="k")

cax1 = make_axes_locatable(ax1).append_axes('bottom', size='5%', pad=.5)
cax2 = make_axes_locatable(ax2).append_axes('bottom', size='5%', pad=.5)
cax3 = make_axes_locatable(ax3).append_axes('bottom', size='5%', pad=.5)
fig.colorbar(im1, cax=cax1, orientation="horizontal")
fig.colorbar(im2, cax=cax2, orientation="horizontal")
fig.colorbar(im3, cax=cax3, orientation="horizontal")
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_ylabel("time")
ax2.set_ylabel("time")
ax3.set_ylabel("time")
ax1.set_xlabel("x")
ax2.set_xlabel("x")
ax3.set_xlabel("x")
plt.suptitle("The ROM is based on 10 FEM simulations (o) and can approximate solutions for any time")

ax1.set_title("Exact solution")
ax2.set_title("exact solution projected into reduced space and back")
ax3.set_title("interpolated solution")
plt.show()


fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = VT_approx_fine.shape
h, w = (times.max() - times.min())/ncols, (x[-1]-x[0])/nrows
left, right, bottom, top = x[0]-h/2, x[-1]+h/2, times.min()-w/2, times.max()+w/2
im1 = ax1.imshow(VTe_fine, cmap=cmap, vmin=VT.min(), vmax=VT.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
# im2 = ax2.imshow(VT_proj_fine.T, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
#                  interpolation='nearest', extent=(left, right, bottom, top))
im3 = ax3.imshow(VT_approx_fine, cmap=cmap, vmin=VT.min(), vmax=VT.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))

# for i in range(len(times)):
#     color = X[:, i]
#     ax3.scatter(np.ones_like(x) * times[i], x, s=10, c=color, cmap=cmap,
#                 vmin=X.min(), vmax=X.max(), edgecolor="k")

cax1 = make_axes_locatable(ax1).append_axes('bottom', size='5%', pad=.5)
# cax2 = make_axes_locatable(ax2).append_axes('bottom', size='5%', pad=.5)
cax3 = make_axes_locatable(ax3).append_axes('bottom', size='5%', pad=.5)
fig.colorbar(im1, cax=cax1, orientation="horizontal")
# fig.colorbar(im2, cax=cax2, orientation="horizontal")
fig.colorbar(im3, cax=cax3, orientation="horizontal")
ax1.set_aspect('auto')
# ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_ylabel("time")
# ax2.set_ylabel("time")
ax3.set_ylabel("time")
ax1.set_xlabel("x")
# ax2.set_xlabel("x")
ax3.set_xlabel("x")
plt.suptitle("The ROM is based on 10 FEM simulations (o) and can approximate solutions for any time")

ax1.set_title("Exact solution")
# ax2.set_title("exact solution projected into reduced space and back")
ax3.set_title("interpolated solution")
plt.show()



# plot bsp
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54))
ax1.plot(x, X[:, 3], color='#1f77b4', marker=".")
ax1.plot(x, X[:, 4], color='#ff7f0e', marker=".")
ax1.plot(x, Xe_proj_fine[:, 35], color='#2ca02c', marker=".")
ax2.plot(np.arange(len(VT[:, 0]))+1, VT[:, 3], color='#1f77b4', marker=".")
ax2.plot(np.arange(len(VT[:, 0]))+1, VT[:, 4], color='#ff7f0e', marker=".")
ax2.plot(np.arange(len(VTe_fine[:, 0]))+1, VTe_fine[:, 35], color='#2ca02c', marker=".")
# ax2.plot(np.arange(0, r), np.cumsum(S)/S.sum(), "bo")
# ax.plot(x[0], Xe[:, j], "g.", label="exact solution")
ax1.set_xlabel("x")
ax1.set_ylabel("u(x)")
ax2.set_xlabel("mode #")
ax2.set_ylabel("Right singular value")
# plt.title("dacay of singular values")
# ax1.set_xlim([0, r])
plt.show()



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(Uhat, vmin=-.01, vmax=.01, interpolation="nearest")
ax1.set_aspect("auto")
ax2.imshow(U_fourier, vmin=-.01, vmax=.01, interpolation="nearest")
ax2.set_aspect("auto")
ax3.imshow(U_fourier, vmin=-.01, vmax=.01, interpolation="nearest")
ax3.set_aspect("auto")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(np.matmul(Uhat.T, X), interpolation="nearest")  # VThat
ax1.set_aspect("auto")
ax2.imshow(np.matmul(U_fourier.T, X), interpolation="nearest")
ax2.set_aspect("auto")
ax3.imshow(np.matmul(U_fourier2.T, X), interpolation="nearest")
ax3.set_aspect("auto")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(np.matmul(Uhat, np.matmul(Uhat.T, X)), vmin=0, vmax=1, interpolation="nearest")
ax1.set_aspect("auto")
ax2.imshow(np.matmul(U_fourier, np.matmul(U_fourier.T, X)), vmin=0, vmax=1, interpolation="nearest")
ax2.set_aspect("auto")
ax3.imshow(np.matmul(U_fourier2, np.matmul(U_fourier2.T, X)), vmin=0, vmax=1, interpolation="nearest")
ax3.set_aspect("auto")
plt.show()




# n: Window length.
# dx: Sample spacing (inverse of the sampling rate). Defaults to 1.
# d = 1.0
val = 1.0 / (n * dx)  # smallest frequency
results = np.empty(n, int)
N = (n-1)//2 + 1
p1 = np.arange(0, N, dtype=int)
results[:N] = p1
p2 = np.arange(-(n//2), 0, dtype=int)
results[N:] = p2
f = results * val


i_sample = n//2
x = x  # time
signal = X[:, i_sample]
fourier = np.fft.fft(signal)
# n = signal.size
timestep = dx
freq = np.fft.fftfreq(m, d=timestep)
# np.fft.ifft


# X[:, 3] = np.matmul(U*S, VT[:, 3])
# np.matmul(U.T, X[:, 3]) = VT[:, 3]*S
# np.matmul(Uhat.T, X[:, 3]) = VThat[:, 3]*Shat
y = fft(X[:, i_sample])

coeff = VThat[:, i_sample]*Shat  # k
coeffe = (VT[:, i_sample]*S)[:reduced_rank]

# coeff1 = (np.matmul(U.T, X[:, 30]))[:reduced_rank]
coeff1 = np.matmul((U_fourier.T)[:, :], X[:, i_sample])
coeff2 = np.matmul(U_fourier2.T, X[:, i_sample])


N = int(n/2+1)
fa = 1.0/dx # scan frequency
f_f = np.linspace(0, fa/2, N, endpoint=True)

def get_empirical_basis(X):
    m, n = X.shape
    a = X.mean(axis=1)
    U = np.zeros((m, n))
    t = np.linspace(0, 1, m, endpoint=False)
    # t = np.linspace(0, 1, len(x))
    for i in range(n):
        U [:, j] = A*np.sin(2*np.pi *(j+1)/2*t)
        # i = 0
        w = 2*np.pi *(i+1)/2
        U[i] = np.cos(2*np.pi *(i+1)/2*t)
    return U



def get_f_basis(signal, t):
    N = len(signal)
    freq = np.fft.fftfreq(N, d=t[1]-t[0])
    r = np.sum(freq>0)
    U = np.zeros((N, 2*r))
    w = np.zeros((2*r))
    for n in range(1, r+1):
        w_ = freq[n]*2*np.pi
        # print("", freq[n]*2*np.pi, -2*dt/T * A1[n].imag, 2*dt/T * A1[n].real, sep="\t")
        U[:, 2*n-2] = np.sin(w_*t)  # n*pi flips the sign for odd n
        U[:, 2*n-1] = np.cos(w_*t)    
        w[2*n-2] = w[2*n-1] = w_
    U = U / np.sum(U**2, axis=0)**.5
    VT = np.matmul(U.T, signal)
    # g1_approx = np.matmul(U, VT)
    S = (VT*VT).sum()**.5  # its only 1 mode
    return signal.mean(), U, S, VT/S, w

m, Uf, Sf, VTf, ff = get_f_basis(signal, x)
coeff_f = (VTf*Sf)
coeff_f[coeff_f<1e-6] = 1e-6
coeff[coeff<1e-6] = 1e-6
coeff1[coeff1<1e-6] = 1e-6
coeff2[coeff2<1e-6] = 1e-6

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(2*plot_width/2.54, plot_width/2.54))
# ax.plot(freq, np.abs(fourier), "ko", label="fourier basis")
ax.plot(ff, coeff_f, "ko", label="fourier basis")
ax.plot(f_svd, np.abs(coeff), "ro", label="SVD basis")
# ax.plot(f_svd, np.abs(coeffe), "k.", label="SVD basis")
# ax.plot(f_svd, np.abs(coeff1), "g.", label="SVD basis (sine functions)")
# ax.plot(f_svd2/(2*L), np.abs(coeff2), "b.", label="SVD basis (empirical)")
ax.set_xlabel("frequency")
ax.set_ylabel("coefficient for snapshot at t=0.35")
ax.legend()
ax.set_xlim([0, 15])
ax.set_yscale('log')
# ax2.plot(np.cum(VTf)/np.sum(VTf))
ax2.plot(np.cumsum(np.abs(coeff_f))/np.sum(np.abs(coeff_f)), "k.", label="fourier basis")
ax2.plot(np.cumsum(np.abs(coeff))/np.sum(np.abs(coeff)), "ro", label="SVD basis")
# ax.plot(f_svd, np.abs(coeffe), "k.", label="SVD basis")
# ax2.plot(np.cumsum(np.abs(coeff1))/np.sum(np.abs(coeff1)), "go", label="SVD basis (sine functions)")
# ax2.plot(np.cumsum(np.abs(coeff2))/np.sum(np.abs(coeff2)), "bo", label="SVD basis (empirical)")
# ax1.set_xlim([0, 15])
plt.show()

ff[coeff_f>1e-5]
f_svd[coeff>1e-5]
f_svd[coeff1>1e-5]
(f_svd2/(2*L))[coeff2>1e-5]


e0 = np.std(X - np.matmul(U*S, VT))
e1 = np.std(X - np.matmul(Uhat*Shat, VThat))
e2 = np.std(X - np.matmul(U_fourier*Shat, VThat))
e3 = np.std(X - np.matmul(U_fourier2*Shat, VThat))

T_svd = 1/f_svd*2*np.pi


# frequencies do not match.
# fourier: has mean value as first mode

