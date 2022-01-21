# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:31:49 2021

@author: florianma
"""
from ROM.snapshot_manager import Data, load_snapshots_cavity, load_snapshots_cylinder
from low_rank_model_construction.proper_orthogonal_decomposition import row_svd
import numpy as np

path = "\\\\files.ad.ife.no/MatPro_files/Florian/flow_around_cylinder/"
# path = "\\\\files.ad.ife.no/MatPro_files/Florian/cavity/"

# if "my_data" not in locals():
#     # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
#     X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cylinder(path)
    
#     my_data = Data(X_all, _xi_all_)
#     my_data.normalise()
#     X = my_data.X_n
X = np.load("C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Desktop/Xn_cyl.npy")#.astype(np.float32)
a = np.arange(len(X[0]))
np.random.shuffle(a)
# U, S, VT = row_svd(X[:, a], 16, eps=1.0-1E-5,
#                    ommit_V=True, QR_DECOMPOSITION=True)
X_rand = X[:, a]
U, S, VT = row_svd(X_rand, 8, eps="5000fixed",
                   ommit_V=True, QR_DECOMPOSITION=True)
if False:
    np.save("\\\\files.ad.ife.no/MatPro_files/Florian/flow_around_cylinder/Xn_S", S)
    np.save("\\\\files.ad.ife.no/MatPro_files/Florian/flow_around_cylinder/Xn_U", U)
    # np.save("\\\\files.ad.ife.no/MatPro_files/Florian/flow_around_cylinder/Xn_VT", VT)
    np.save("C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Desktop/X_rand", X_rand)
    del my_data
