# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:31:49 2021

@author: florianma
"""
from ROM.snapshot_manager import Data, load_snapshots_cavity, load_snapshots_cylinder
from low_rank_model_construction.proper_orthogonal_decomposition import row_svd
import numpy as np
from scipy.interpolate import interp1d

path = "\\\\files.ad.ife.no/MatPro_files/Florian/CavitySolidification/"
# path = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/"

def rho_(T):
    t = np.array([0.00, 1000])
    r = np.array([2.0, 1.0])
    f_rho = interp1d(t, r, kind='linear', bounds_error=False,
                     fill_value="extrapolate")
    return f_rho(T)  # kg/m3


def mu_(T, m=10):
    if isinstance(T, int) or isinstance(T, float):
        T = [T]
    mu_liquidus = 1  # water
    mu = np.array((T-650)**2*m)+mu_liquidus
    mu[T>650] = mu_liquidus
    return mu

if False:
    prefix = "mul1e6/"
    uv = np.load(path+prefix+"Tamb570_velocity.npy")  # (5000, 6026)
    p = np.load(path+prefix+"Tamb570_pressure.npy")  # (5000, 3013)
    t = np.load(path+prefix+"Tamb570_temperature.npy")
    r = rho_(t)
    m = mu_(t)
    
    X = np.concatenate((uv.T, p.T, t.T), axis=0) #  X = p.T
    my_data = Data(X, False)
    X_n = my_data.normalise()
    X_n[X_n<.0001] = 0.0
    # path = "\\\\files.ad.ife.no/MatPro_files/Florian/flow_around_cylinder/"
    # path = "\\\\files.ad.ife.no/MatPro_files/Florian/cavity/"
    
    # if "my_data" not in locals():
    #     # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
    #     X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cylinder(path)
        
    #     my_data = Data(X_all, _xi_all_)
    #     my_data.normalise()
    #     X = my_data.X_n
    # X = np.load("C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Desktop/Xn_cyl.npy")#.astype(np.float32)
    
    a = np.arange(len(X_n[0]))
    np.random.shuffle(a)
    # U, S, VT = row_svd(X[:, a], 16, eps=1.0-1E-5,
    #                    ommit_V=True, QR_DECOMPOSITION=True)
    X_rand = X_n[:, a]
    U, S, VT = row_svd(X_rand, 1, 1, False, False)
    # U, S, VT = row_svd(X_rand, 5, eps="1000fixed",
    #                    ommit_V=True, QR_DECOMPOSITION=True)
    np.save(path+prefix+"Xn_S", S)
    np.save(path+prefix+"Xn_U", U)
    # np.save(path+"Xn_VT", VT)
    # np.save("C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Desktop/X_rand", X_rand)
    # del my_data

# asd
import matplotlib.pyplot as plt
plot_width = 8
energy = True

multiplicator = [1/100, 1/10, 1, 10, 100, 1000, 10000, 100000, 1e6]
T = np.linspace(570, 670, 10000)
fig, (ax1) = plt.subplots(1, 1, figsize=(plot_width/2.54*2, plot_width/2.54))
for m in multiplicator:
    ax1.plot(T, mu_(T, m), label="mu=(T-650)**2*{:.0e})+1".format(m))
# ax1.set_yscale('log')
ax1.set_xlim([570, 670])
ax1.set_ylim([0, 1000])
ax1.set_xlabel("temperature")
ax1.set_ylabel("viscosity")
plt.legend()
plt.tight_layout()
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(plot_width/2.54*2, plot_width/2.54))
for m in multiplicator:
    ax1.plot(T, mu_(T, m), label="mu=(T-650)**2*{:.0e})+1".format(m))
ax1.set_xlim([640, 660])
ax1.set_ylim([0, 100])
plt.legend()
plt.show()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
prefixes = ["div100_non_solid", "div10", "ref", "mul10", "mul100", "mul1000", "mul10000", "mul100000", "mul1000000", "mul1e6"]
multiplicator = [1/100, 1/10, 1, 10, 100, 1000, 10000, 100000, 1000000, 1e6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_width/2.54*2, plot_width/2.54))
# for S, col, lbl in zip([S1, S2], ['#1f77b4', '#ff7f0e'], ["w mushy zone", "w/o mushy zone"]):
for i, prefix in enumerate(prefixes):
    # i = 2
    print(i, prefix)
    # prefix = "ref"
    # prefix = prefixes[i]
    col = colors[i]
    S = np.load(path + prefix + "/Xn_S.npy")#[25:200]
    print(len(S), S[::100])
    lbl = "mu=(T-650)**2*{:.0e})+1".format(multiplicator[i])
    ax1.plot(np.arange(0, len(S)), S, color=col, marker=".")
    ax2.plot(np.arange(0, len(S)), np.cumsum(S)/S.sum()*100,
              color=col, marker=".", label=lbl)
ax1.set_xlabel("rank r")
ax1.set_ylabel("singular values")
ax2.set_xlabel("rank r")
ax1.set_yscale('log')
ax1.set_ylim([1e-6, 1e3])
ax2.set_ylim([95, 100])
ax2.set_ylabel("Cumulative Energy [%]")
ax2.legend()
ax1.grid(which="both")
# plt.title("dacay of singular values")
ax1.set_xlim([0, 1000])
ax2.set_xlim([0, 200])
plt.grid()
plt.tight_layout()
plt.show()
