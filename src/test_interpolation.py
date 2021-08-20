# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:38:08 2021

@author: florianma
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import timeit
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf, griddata
# from amyavi import mlab
plot_width = 8
plt.close("all")

x1_train = np.load("C:/Users/florianma/Documents/data/testRBF/x1_train.npy")
x2_train = np.load("C:/Users/florianma/Documents/data/testRBF/x2_train.npy")
V0 = np.load("C:/Users/florianma/Documents/data/testRBF/V0.npy")
V1 = np.load("C:/Users/florianma/Documents/data/testRBF/V1.npy")
V2 = np.load("C:/Users/florianma/Documents/data/testRBF/V2.npy")
V0[V0 < 1e-6] = 0
V1[V1 < 1e-6] = 0
V2[V2 < 1e-6] = 0
V = [V0, V1, V2]
# d = np.load("C:/Users/florianma/Documents/data/testRBF/d_train.npy")
x1_validation = np.load(
    "C:/Users/florianma/Documents/data/testRBF/x1_validation.npy")
x2_validation = np.load(
    "C:/Users/florianma/Documents/data/testRBF/x2_validation.npy")

# V[0][250, 0 ] = 1
# V[1][250, 0 ] = 1
# V[2][250, 0 ] = 1
n_modes = 1
datasets, snapshots_per_dataset3 = x1_train.shape
snapshots_per_dataset = snapshots_per_dataset3//3
V_interpolated = 3*[np.zeros((snapshots_per_dataset, n_modes))]
for j in range(3):
    for k in range(n_modes):
        d_ = V[j][:, k].reshape(datasets, snapshots_per_dataset).copy()
        d = np.c_[d_, d_, d_]  # 8, 3*150 repeats in each oscillation
        # radial basis function interpolator instance
        # rbfi = Rbf(x1_train.ravel(), x2_train.ravel(), d.ravel())
        # interpolators[j][k] = rbfi
        points = np.c_[x1_train.ravel(), x2_train.ravel()]
        di = griddata(points=points.copy(), values=d.ravel().copy(),
                      xi=np.c_[x1_validation.copy(), x2_validation.copy()], method='nearest')
        # rbfi = interpolators[j][k]
        # di = rbfi(x1_validation, x2_validation)
        V_interpolated[j][:, k] = di.copy()
        if k == 0:
            print(x1_validation[0], x1_train.ravel()[5*3*50+50+0])
            print(x1_validation[15], x1_train.ravel()[5*3*50+50+15])
            print(x2_validation[0], x2_train.ravel()[5*3*50+50+0])
            print(x2_validation[15], x2_train.ravel()[5*3*50+50+15])
            print()
            print(di[0]-d_.ravel()[5*50+0])
            print(di[0]-d.ravel()[5*3*50+50+0])
            print(di[15]-d.ravel()[5*3*50+50+15])

            print(V_interpolated[j][0, k], V[j][0, k])
            print(V_interpolated[j][0, k], V[j][250, k])
            for s in range(15):
                print(np.where(V_interpolated[j][s, k]-V[j][:, k] == 0))
            print(np.allclose(V_interpolated[j]
                  [0, k], V[j][250, k], atol=1e-6))
fig, ax = plt.subplots()
plt.scatter(x1_train, x2_train, s=15, c=d,
            vmin=.022423928378509626, vmax=0.02599471815644643)
# plt.scatter(x1_train, x2_train, s=15, c=d)
plt.scatter(x1_validation*0+0.0011, x2_validation, s=15, c=di,
            vmin=.022423928378509626, vmax=0.02599471815644643)
V_interpolated[j][0, :]
V[j][300, :n_modes]

# asd


fig, ax = plt.subplots(figsize=(plot_width/2.54, 8/2.54))
plt.plot(x1_train, x2_train, "g.", label="trainings data")
plt.plot(x1_validation, x2_validation, "r.", label="validation data")
plt.legend()
plt.xlabel("viscosity")
plt.ylabel("time")
plt.title("interpolation domain")
plt.tight_layout()
plt.show()

interpolator = Rbf(x1_train.ravel(), x2_train.ravel(), d.ravel(),
                   function="quintic")
di = interpolator(x1_validation, x2_validation)
print(di)
print(di.min(), di.max(), di.mean())
print(d.min(), d.max(), d.mean())

points = np.c_[x1_train.ravel(), x2_train.ravel()]
# values = d
# xi = np.c_[x1_validation, x2_validation]
x1_validation = x1_validation*0 + 0.0008
grid_z1 = griddata(points, d.ravel(), (x1_validation, x2_validation),
                   method='linear')
grid_z2 = griddata(points, d.ravel(), (x1_validation, x2_validation),
                   method='cubic')
print(grid_z1.min(), grid_z1.max(), grid_z1.mean())
print(grid_z2.min(), grid_z2.max(), grid_z2.mean())
l = x1_train.ravel() == 0.0008
print(d.ravel()[l])
print(grid_z1)
V.t[:, 150]

if False:
    np.save("C:/Users/florianma/Documents/data/testRBF/x1_train.npy", x1_train)
    np.save("C:/Users/florianma/Documents/data/testRBF/x2_train.npy", x2_train)
    np.save("C:/Users/florianma/Documents/data/testRBF/d_train.npy", d)
    np.save("C:/Users/florianma/Documents/data/testRBF/x1_validation.npy", x1_validation)
    np.save("C:/Users/florianma/Documents/data/testRBF/x2_validation.npy", x2_validation)
    np.save("C:/Users/florianma/Documents/data/testRBF/V0.npy", V[0].numpy())
    np.save("C:/Users/florianma/Documents/data/testRBF/V1.npy", V[1].numpy())
    np.save("C:/Users/florianma/Documents/data/testRBF/V2.npy", V[2].numpy())
