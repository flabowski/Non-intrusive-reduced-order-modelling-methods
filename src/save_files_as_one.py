#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:31:33 2021

@author: florianma
"""
import numpy as np
import os

N = 106959
J = 10000
path = '/home/florianma@ad.ife.no/Documents/NIROM/doc/'
x = np.load(path+"{:06.0f}x.npy".format(0))
y = np.load(path+"{:06.0f}y.npy".format(0))
tri = np.load(path+"{:06.0f}t.npy".format(0))
nodes = len(x)

u = np.zeros((J, nodes))
v = np.zeros((J, nodes))
p = np.zeros((J, nodes))
j = 0
for n in range(50001, N):
    if ((n % J) < 1e-4):
        print(n, end=" ")
        np.save(path+"{:06.0f}to{:06.0f}_u.npy".format(n-J, n), u)
        np.save(path+"{:06.0f}to{:06.0f}_v.npy".format(n-J, n), v)
        np.save(path+"{:06.0f}to{:06.0f}_p.npy".format(n-J, n), p)
        j = 0
    u[j] = np.load(path+"{:06.0f}u.npy".format(n+1))
    v[j] = np.load(path+"{:06.0f}v.npy".format(n+1))
    p[j] = np.load(path+"{:06.0f}p.npy".format(n+1))
    j += 1
np.save(path+"{:06.0f}to{:06.0f}_u.npy".format((n//J)*J, n), u)
np.save(path+"{:06.0f}to{:06.0f}_v.npy".format((n//J)*J, n), v)
np.save(path+"{:06.0f}to{:06.0f}_p.npy".format((n//J)*J, n), p)

np.save(path+"x.npy", x)
np.save(path+"y.npy", y)
np.save(path+"tri.npy", tri)

for n in range(50001, N):
    os.remove(path+"{:06.0f}u.npy".format(n+1))
    os.remove(path+"{:06.0f}v.npy".format(n+1))
    os.remove(path+"{:06.0f}p.npy".format(n+1))
# for filename in os.listdir():
#     if filename.endswith(".npy"):
#          # print(os.path.join(directory, filename))
#         continue
#     else:
#         continue
