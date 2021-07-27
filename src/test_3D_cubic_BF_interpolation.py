# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:50:28 2021

@author: florianma
"""
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf, NdPPoly, lagrange)  # RBFInterpolator  NdGridSplinePPForm
import numpy as np
from nirom.src.cross_validation import load_snapshots_cavity, plot_snapshot_cav
from nirom.low_rank_model_construction.proper_orthogonal_decomposition import Data
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
plot_width = 16
LINALG_LIB = "tensorflow"
timed = True


n1, n2, n3 = 8, 5, 7
x1 = np.linspace(0, 8, n1)
x2 = np.linspace(100, 500, n2)
x3 = np.linspace(10, 16, n3)
grid = (x1, x2, x3)
xx, yy, zz = np.meshgrid(*grid, indexing="ij")
f = np.random.rand(*xx.shape)
print(f.shape)

n = 4
n1_f = (n1-1)*n+1
n2_f = (n2-1)*n+1
n3_f = (n2-1)*n+1
x1_f = np.linspace(0, 8, n1_f)
x2_f = np.linspace(100, 500, n2_f)
x3_f = np.linspace(10, 16, n3_f)
xx_f, yy_f, zz_f = np.meshgrid(x1_f, x2_f, x3_f, indexing="ij")
xi_new = np.array([xx_f.ravel(), yy_f.ravel(), zz_f.ravel()]).T  # 3, 495000
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # CUBIC BASIS FUNCTION INTERPOLATION ON A HYPER CUBE # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
grid, f, xi_new
dims = len(grid)
for grd in grid:
    assert len(
        grd) >= 2, "need at least 3 points along each dimension for cubic interpolation."
assert xi_new.shape[1] == dims, "xi_new needs to be shaped (-1, {:.0f})".format(
    dims)

# 1. find indices for hyper rectangle
rgi = RegularGridInterpolator(grid, f, bounds_error=False)
indices1, norm_distances, out_of_bounds = rgi._find_indices(xi_new.T)

# OPTION 1 (simple)


def lagrangepolynomial(p, i, x):
    # 3 pkt in 2 dimensionen: P.shape=(3, 2)
    n = len(p)
    res = 1.0
    for j in range(n):
        if j != i:
            res *= (x-p[j])/(p[i]-p[j])
    return res


def plot_lagrange_poly():
    x = np.linspace(0, 1, 100)
    y0 = lagrangepolynomial([0.0, 0.5, 1.0], 0, x)
    x = np.linspace(0, 1, 100)
    y0 = lagrangepolynomial([0.0, 0.5, 1.0], 0, x)
    plt.plot(x, y0)
    y1 = lagrangepolynomial([0.0, 0.5, 1.0], 1, x)
    plt.plot(x, y1)
    y2 = lagrangepolynomial([0.0, 0.5, 1.0], 2, x)
    plt.plot(x, y2)
    plt.show()


t0 = timeit.default_timer()

# # indices1 contains the an index for every point. this index points to the next grid point left of that
# # since we have 3 points per dimension for the cubic interpolation, we want to use abother point left of that (given it exists)
# ind_grid = [indices1[i].copy() for i in range(len(indices1))]
# for d in range(dims):
#     is_not_on_left_border = indices1[d] != 0
#     ind_grid[d][is_not_on_left_border] -= 1

# # actual interpolation, here 3D
f_interpolated = np.zeros(len(xi_new))
# for i, point in enumerate(xi_new):
#     # index for left side of the hyper cube for each dimension
#     lower_i = [ind_grid[d][i] for d in range(dims)]
#     # p = np.zeros((3**dims, dims))
#     px = [grid[0][lower_i[0]], grid[0][lower_i[0]+1], grid[0][lower_i[0]+2]]
#     lpx0 = lagrangepolynomial(px, 0, point[0])
#     lpx1 = lagrangepolynomial(px, 1, point[0])
#     lpx2 = lagrangepolynomial(px, 2, point[0])

#     py = [grid[1][lower_i[1]], grid[1][lower_i[1]+1], grid[1][lower_i[1]+2]]
#     lpy0 = lagrangepolynomial(py, 0, point[1])
#     lpy1 = lagrangepolynomial(py, 1, point[1])
#     lpy2 = lagrangepolynomial(py, 2, point[1])

#     pz = [grid[2][lower_i[2]], grid[2][lower_i[2]+1], grid[2][lower_i[2]+2]]
#     lpz0 = lagrangepolynomial(pz, 0, point[2])
#     lpz1 = lagrangepolynomial(pz, 1, point[2])
#     lpz2 = lagrangepolynomial(pz, 2, point[2])

#     for d1, lpx in enumerate([lpx0, lpx1, lpx2]):
#         for d2, lpy in enumerate([lpy0, lpy1, lpy2]):
#             for d3, lpz in enumerate([lpz0, lpz1, lpz2]):
#                 f_interpolated[i] += lpx*lpy*lpz * f[lower_i[0]+d1,
#                                                       lower_i[1]+d2,
#                                                       lower_i[2]+d3]

t1 = timeit.default_timer()


# OPTION 2: define the basis function beforehand

def cubic_coeff(p, i):
    assert len(p) == 3, "need 3 points for cubic basis function"
    yi = p[i]
    if i == 0:
        y1, y2 = p[1], p[2]
    elif i == 1:
        y1, y2 = p[0], p[2]
    elif i == 2:
        y1, y2 = p[0], p[1]
    else:
        raise ValueError("i must be 0, 1 or 2")
    n = (yi-y1)*(yi-y2)
    return (1.0/n, -(y1+y2)/n, y1*y2/n)


def test_cubic_coeff():
    x0, x1, x2 = 0, .5, 2.0
    a, b, c = cubic_coeff([x0, x1, x2], 0)
    assert (a*x0**2 + b*x0 + c) == 1
    assert (a*x1**2 + b*x1 + c) == 0
    assert (a*x2**2 + b*x2 + c) == 0

    a, b, c = cubic_coeff([x0, x1, x2], 1)
    assert (a*x0**2 + b*x0 + c) == 0
    assert (a*x1**2 + b*x1 + c) == 1
    assert (a*x2**2 + b*x2 + c) == 0

    a, b, c = cubic_coeff([x0, x1, x2], 2)
    assert (a*x0**2 + b*x0 + c) == 0
    assert (a*x1**2 + b*x1 + c) == 0
    assert (a*x2**2 + b*x2 + c) == 1
    return


def get_ind_element(ind_pt):
    # there are 3 nodes per element, while the side nodes are shared
    return int(ind_pt//2)


def get_BF_coefficients_along(xi):
    # basis functions. shaped i, j, d
    # i: global point index (0....n)
    # all points within one rectangle have the same values!
    # j: local point in rectangle index (0, 1, 2)
    # 3: the 3 coefficients (a, b, c) of the polynom a*x**2 + b*x + c
    # note: if the number of points is even, an additional rectangle is added
    # using the points n, n-1, and n-2, even though n-1 and n-2 are used for
    # the second last rectangle too
    print(xi)
    n_elements = get_ind_element(ind_pt=len(xi))
    Phi_i = np.zeros((n_elements, 3, 3))
    ind_grd = np.zeros((n_elements), dtype=np.int64)
    ind_element = 0
    for i in range(0, len(xi)-1, 2):
        if (i+3) > len(xi):
            p = xi[i-1:i+2]
            ind_grd[ind_element] = i-1
        else:
            p = xi[i:i+3]
            ind_grd[ind_element] = i
        print(p)
        Phi_i[ind_element, 0] = cubic_coeff(p, 0)
        Phi_i[ind_element, 1] = cubic_coeff(p, 1)
        Phi_i[ind_element, 2] = cubic_coeff(p, 2)
        ind_element += 1
    return Phi_i, ind_grd


t2 = timeit.default_timer()


Phi = [None for d in range(dims)]
ind_grd = [None for d in range(dims)]
for d in range(dims):
    Phi[d], ind_grd[d] = get_BF_coefficients_along(grid[d])

# # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # #
fig, ax = plt.subplots()
for elem in range(len(Phi[0])):
    a, b, c = Phi[0][elem, 0, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
    a, b, c = Phi[0][elem, 1, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
    a, b, c = Phi[0][elem, 2, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
plt.ylim(-.2, 1)
plt.show()
# # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # #


# TODO: enable broadcasting for f
f_interpolated2 = np.zeros(len(xi_new))
for i, point in enumerate(xi_new):
    grid_index_left = [None for d in range(dims)]
    # j: BFx, BFy, BFz, ..., BFn
    BF = [[None for j in range(3)] for d in range(dims)]

    # find indices for hyperrectangle
    # index for left side of the hyper cube for each dimension
    # lower_i = [ind_grid[d][i] for d in range(dims)]
    d = 0
    for d in range(dims):
        # BF[d] is lpx, lpy, lpz
        # Phi[d] is Phix, Phiy, Phiz
        # point[d] is x, y, z
        i_element = get_ind_element(indices1[d][i])
        for j in range(3):
            a, b, c = Phi[d][i_element, j, :]
            BF[d][j] = a*point[d]**2 + b*point[d] + c
        grid_index_left[d] = ind_grd[d][i_element]
    # dim1
    # d = 1
    # i_element = get_ind_element(indices1[d][i])
    # a, b, c = Phi[d][i_element, 0, :]
    # BF[d][0] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 1, :]
    # BF[d][1] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 2, :]
    # BF[d][2] = a*point[d]**2 + b*point[d] + c
    # grid_index_left[d] = ind_grd[d][i_element]
    # # dim2
    # d = 2
    # i_element = get_ind_element(indices1[d][i])
    # a, b, c = Phi[d][i_element, 0, :]
    # BF[d][0] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 1, :]
    # BF[d][1] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 2, :]
    # BF[d][2] = a*point[d]**2 + b*point[d] + c
    # grid_index_left[d] = ind_grd[d][i_element]

    # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # #
    # if i < 20:
    #     print(x, y, z, lpz0+lpz1+lpz2)
    #     fig, ax = plt.subplots()
    #     a, b, c = Phiz[i_element, 0, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     a, b, c = Phiz[i_element, 1, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     a, b, c = Phiz[i_element, 2, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     plt.plot(z, lpz0+lpz1+lpz2, "ro")
    #     plt.show()
    # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # #

    if dims == 1:
        for j1, bfx in enumerate(BF[0]):
            f_interpolated2[i] += bfx * f[grid_index_left[0]+j1]
    if dims == 2:
        for j0, bfx in enumerate(BF[0]):
            for j1, bfy in enumerate(BF[1]):
                f_interpolated2[i] += bfx*bfy * \
                    f[grid_index_left[0]+j0,
                      grid_index_left[1]+j1]
    if dims == 3:
        for j0, bfx in enumerate(BF[0]):
            for j1, bfy in enumerate(BF[1]):
                for j2, bfz in enumerate(BF[2]):
                    f_interpolated2[i] += bfx*bfy*bfz * \
                        f[grid_index_left[0]+j0,
                          grid_index_left[1]+j1,
                          grid_index_left[2]+j2]
    if dims == 4:
        for j0, bf0 in enumerate(BF[0]):
            for j1, bf1 in enumerate(BF[1]):
                for j2, bf2 in enumerate(BF[2]):
                    for j3, bf3 in enumerate(BF[3]):
                        f_interpolated2[i] += bf0*bf1*bf2*bf3 * \
                            f[grid_index_left[0]+j0,
                              grid_index_left[1]+j1,
                              grid_index_left[2]+j2,
                              grid_index_left[3]+j3]
    if dims == 5:
        for j0, bf0 in enumerate(BF[0]):
            for j1, bf1 in enumerate(BF[1]):
                for j2, bf2 in enumerate(BF[2]):
                    for j3, bf3 in enumerate(BF[3]):
                        for j4, bf4 in enumerate(BF[3]):
                            f_interpolated2[i] += bf0*bf1*bf2*bf3*bf4 * \
                                f[grid_index_left[0]+j0,
                                  grid_index_left[1]+j1,
                                  grid_index_left[2]+j2,
                                  grid_index_left[3]+j3,
                                  grid_index_left[4]+j4]
    # ...

t3 = timeit.default_timer()

print(t1-t0)
print(t3-t2)
f_interpolated.shape = (n1_f, n2_f, n3_f)
f_interpolated2.shape = (n1_f, n2_f, n3_f)
# f_interpolated2 = np.transpose(f_interpolated2, (1,0,2))
# print(f_interpolated2.shape)

fig, ax = plt.subplots()
plt.plot(x1, f[:, 0, 0], "go")
plt.plot(x1_f, f_interpolated[:, 0, 0], "r.")
plt.plot(x1_f, f_interpolated2[:, 0, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x2, f[0, :, 0], "go")
plt.plot(x2_f, f_interpolated[0, :, 0], "r.")
plt.plot(x2_f, f_interpolated2[0, :, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x3, f[0, 0, :], "go")
plt.plot(x3_f, f_interpolated[0, 0, :], "r.")
plt.plot(x3_f, f_interpolated2[0, 0, :], "b.")
plt.show()


# j = 0
# for d0 in range(3):  # 2 points along each dimension
#     for d1 in range(3):
#         for d2 in range(3):
#             i0 = lower_i[d0]
#             i1 = lower_i[d1]
#             i2 = lower_i[d2]
#             p[j] = grid[0][lower_i[0]+d0], grid[1][lower_i[1]+d1], grid[2][lower_i[2]+d2]  # ..., grid[n][dn]
#             j +=1
# for j in range(3**dims):
#     f_interpolated[i] += lagrange(p, j, point)
# for d0 in range(2):  # 2 points along each dimension
#     for d1 in range(2):
#         for d2 in range(2):
#             f_interpolated[i] +=

# for lower_i in
# x = np.linspace(0, 1, 100)
# p = lagrange([0, 0.5, 1.0], [1., 0, 0])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# p = lagrange([0, 0.5, 1.0], [0, 1., 0])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# p = lagrange([0, 0.5, 1.0], [0, 0, 1.])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# plt.show()


# rgi(xi_new)
# c =
# ndp = NdPPoly(c, x)


# rbs = RectBivariateSpline(x1, x2, f)
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:50:28 2021

@author: florianma
"""
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf, NdPPoly, lagrange)  # RBFInterpolator  NdGridSplinePPForm
import numpy as np
from nirom.src.cross_validation import load_snapshots_cavity, plot_snapshot_cav
from nirom.low_rank_model_construction.proper_orthogonal_decomposition import Data
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
plot_width = 16
LINALG_LIB = "tensorflow"
timed = True


n1, n2, n3 = 8, 5, 7
x1 = np.linspace(0, 8, n1)
x2 = np.linspace(100, 500, n2)
x3 = np.linspace(10, 16, n3)
grid = (x1, x2, x3)
xx, yy, zz = np.meshgrid(*grid, indexing="ij")
f = np.random.rand(*xx.shape)
print(f.shape)

n = 4
n1_f = (n1-1)*n+1
n2_f = (n2-1)*n+1
n3_f = (n2-1)*n+1
x1_f = np.linspace(0, 8, n1_f)
x2_f = np.linspace(100, 500, n2_f)
x3_f = np.linspace(10, 16, n3_f)
xx_f, yy_f, zz_f = np.meshgrid(x1_f, x2_f, x3_f, indexing="ij")
xi_new = np.array([xx_f.ravel(), yy_f.ravel(), zz_f.ravel()]).T  # 3, 495000
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # CUBIC BASIS FUNCTION INTERPOLATION ON A HYPER CUBE # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
grid, f, xi_new
dims = len(grid)
for grd in grid:
    assert len(
        grd) >= 2, "need at least 3 points along each dimension for cubic interpolation."
assert xi_new.shape[1] == dims, "xi_new needs to be shaped (-1, {:.0f})".format(
    dims)

# 1. find indices for hyper rectangle
rgi = RegularGridInterpolator(grid, f, bounds_error=False)
indices1, norm_distances, out_of_bounds = rgi._find_indices(xi_new.T)

# OPTION 1 (simple)


def lagrangepolynomial(p, i, x):
    # 3 pkt in 2 dimensionen: P.shape=(3, 2)
    n = len(p)
    res = 1.0
    for j in range(n):
        if j != i:
            res *= (x-p[j])/(p[i]-p[j])
    return res


def plot_lagrange_poly():
    x = np.linspace(0, 1, 100)
    y0 = lagrangepolynomial([0.0, 0.5, 1.0], 0, x)
    x = np.linspace(0, 1, 100)
    y0 = lagrangepolynomial([0.0, 0.5, 1.0], 0, x)
    plt.plot(x, y0)
    y1 = lagrangepolynomial([0.0, 0.5, 1.0], 1, x)
    plt.plot(x, y1)
    y2 = lagrangepolynomial([0.0, 0.5, 1.0], 2, x)
    plt.plot(x, y2)
    plt.show()


t0 = timeit.default_timer()

# # indices1 contains the an index for every point. this index points to the next grid point left of that
# # since we have 3 points per dimension for the cubic interpolation, we want to use abother point left of that (given it exists)
# ind_grid = [indices1[i].copy() for i in range(len(indices1))]
# for d in range(dims):
#     is_not_on_left_border = indices1[d] != 0
#     ind_grid[d][is_not_on_left_border] -= 1

# # actual interpolation, here 3D
f_interpolated = np.zeros(len(xi_new))
# for i, point in enumerate(xi_new):
#     # index for left side of the hyper cube for each dimension
#     lower_i = [ind_grid[d][i] for d in range(dims)]
#     # p = np.zeros((3**dims, dims))
#     px = [grid[0][lower_i[0]], grid[0][lower_i[0]+1], grid[0][lower_i[0]+2]]
#     lpx0 = lagrangepolynomial(px, 0, point[0])
#     lpx1 = lagrangepolynomial(px, 1, point[0])
#     lpx2 = lagrangepolynomial(px, 2, point[0])

#     py = [grid[1][lower_i[1]], grid[1][lower_i[1]+1], grid[1][lower_i[1]+2]]
#     lpy0 = lagrangepolynomial(py, 0, point[1])
#     lpy1 = lagrangepolynomial(py, 1, point[1])
#     lpy2 = lagrangepolynomial(py, 2, point[1])

#     pz = [grid[2][lower_i[2]], grid[2][lower_i[2]+1], grid[2][lower_i[2]+2]]
#     lpz0 = lagrangepolynomial(pz, 0, point[2])
#     lpz1 = lagrangepolynomial(pz, 1, point[2])
#     lpz2 = lagrangepolynomial(pz, 2, point[2])

#     for d1, lpx in enumerate([lpx0, lpx1, lpx2]):
#         for d2, lpy in enumerate([lpy0, lpy1, lpy2]):
#             for d3, lpz in enumerate([lpz0, lpz1, lpz2]):
#                 f_interpolated[i] += lpx*lpy*lpz * f[lower_i[0]+d1,
#                                                       lower_i[1]+d2,
#                                                       lower_i[2]+d3]

t1 = timeit.default_timer()


# OPTION 2: define the basis function beforehand

def cubic_coeff(p, i):
    assert len(p) == 3, "need 3 points for cubic basis function"
    yi = p[i]
    if i == 0:
        y1, y2 = p[1], p[2]
    elif i == 1:
        y1, y2 = p[0], p[2]
    elif i == 2:
        y1, y2 = p[0], p[1]
    else:
        raise ValueError("i must be 0, 1 or 2")
    n = (yi-y1)*(yi-y2)
    return (1.0/n, -(y1+y2)/n, y1*y2/n)


def test_cubic_coeff():
    x0, x1, x2 = 0, .5, 2.0
    a, b, c = cubic_coeff([x0, x1, x2], 0)
    assert (a*x0**2 + b*x0 + c) == 1
    assert (a*x1**2 + b*x1 + c) == 0
    assert (a*x2**2 + b*x2 + c) == 0

    a, b, c = cubic_coeff([x0, x1, x2], 1)
    assert (a*x0**2 + b*x0 + c) == 0
    assert (a*x1**2 + b*x1 + c) == 1
    assert (a*x2**2 + b*x2 + c) == 0

    a, b, c = cubic_coeff([x0, x1, x2], 2)
    assert (a*x0**2 + b*x0 + c) == 0
    assert (a*x1**2 + b*x1 + c) == 0
    assert (a*x2**2 + b*x2 + c) == 1
    return


def get_ind_element(ind_pt):
    # there are 3 nodes per element, while the side nodes are shared
    return int(ind_pt//2)


def get_BF_coefficients_along(xi):
    # basis functions. shaped i, j, d
    # i: global point index (0....n)
    # all points within one rectangle have the same values!
    # j: local point in rectangle index (0, 1, 2)
    # 3: the 3 coefficients (a, b, c) of the polynom a*x**2 + b*x + c
    # note: if the number of points is even, an additional rectangle is added
    # using the points n, n-1, and n-2, even though n-1 and n-2 are used for
    # the second last rectangle too
    print(xi)
    n_elements = get_ind_element(ind_pt=len(xi))
    Phi_i = np.zeros((n_elements, 3, 3))
    ind_grd = np.zeros((n_elements), dtype=np.int64)
    ind_element = 0
    for i in range(0, len(xi)-1, 2):
        if (i+3) > len(xi):
            p = xi[i-1:i+2]
            ind_grd[ind_element] = i-1
        else:
            p = xi[i:i+3]
            ind_grd[ind_element] = i
        print(p)
        Phi_i[ind_element, 0] = cubic_coeff(p, 0)
        Phi_i[ind_element, 1] = cubic_coeff(p, 1)
        Phi_i[ind_element, 2] = cubic_coeff(p, 2)
        ind_element += 1
    return Phi_i, ind_grd


t2 = timeit.default_timer()


Phi = [None for d in range(dims)]
ind_grd = [None for d in range(dims)]
for d in range(dims):
    Phi[d], ind_grd[d] = get_BF_coefficients_along(grid[d])

# # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # #
fig, ax = plt.subplots()
for elem in range(len(Phi[0])):
    a, b, c = Phi[0][elem, 0, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
    a, b, c = Phi[0][elem, 1, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
    a, b, c = Phi[0][elem, 2, :]
    plt.plot(x1_f, a*x1_f**2+b*x1_f+c)
plt.ylim(-.2, 1)
plt.show()
# # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # #


# TODO: enable broadcasting for f
f_interpolated2 = np.zeros(len(xi_new))
for i, point in enumerate(xi_new):
    grid_index_left = [None for d in range(dims)]
    # j: BFx, BFy, BFz, ..., BFn
    BF = [[None for j in range(3)] for d in range(dims)]

    # find indices for hyperrectangle
    # index for left side of the hyper cube for each dimension
    # lower_i = [ind_grid[d][i] for d in range(dims)]
    d = 0
    for d in range(dims):
        # BF[d] is lpx, lpy, lpz
        # Phi[d] is Phix, Phiy, Phiz
        # point[d] is x, y, z
        i_element = get_ind_element(indices1[d][i])
        for j in range(3):
            a, b, c = Phi[d][i_element, j, :]
            BF[d][j] = a*point[d]**2 + b*point[d] + c
        grid_index_left[d] = ind_grd[d][i_element]
    # dim1
    # d = 1
    # i_element = get_ind_element(indices1[d][i])
    # a, b, c = Phi[d][i_element, 0, :]
    # BF[d][0] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 1, :]
    # BF[d][1] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 2, :]
    # BF[d][2] = a*point[d]**2 + b*point[d] + c
    # grid_index_left[d] = ind_grd[d][i_element]
    # # dim2
    # d = 2
    # i_element = get_ind_element(indices1[d][i])
    # a, b, c = Phi[d][i_element, 0, :]
    # BF[d][0] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 1, :]
    # BF[d][1] = a*point[d]**2 + b*point[d] + c
    # a, b, c = Phi[d][i_element, 2, :]
    # BF[d][2] = a*point[d]**2 + b*point[d] + c
    # grid_index_left[d] = ind_grd[d][i_element]

    # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # #
    # if i < 20:
    #     print(x, y, z, lpz0+lpz1+lpz2)
    #     fig, ax = plt.subplots()
    #     a, b, c = Phiz[i_element, 0, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     a, b, c = Phiz[i_element, 1, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     a, b, c = Phiz[i_element, 2, :]
    #     plt.plot(x3_f, a*x3_f**2+b*x3_f+c)
    #     plt.plot(z, lpz0+lpz1+lpz2, "ro")
    #     plt.show()
    # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # #

    if dims == 1:
        for j1, bfx in enumerate(BF[0]):
            f_interpolated2[i] += bfx * f[grid_index_left[0]+j1]
    if dims == 2:
        for j0, bfx in enumerate(BF[0]):
            for j1, bfy in enumerate(BF[1]):
                f_interpolated2[i] += bfx*bfy * \
                    f[grid_index_left[0]+j0,
                      grid_index_left[1]+j1]
    if dims == 3:
        for j0, bfx in enumerate(BF[0]):
            for j1, bfy in enumerate(BF[1]):
                for j2, bfz in enumerate(BF[2]):
                    f_interpolated2[i] += bfx*bfy*bfz * \
                        f[grid_index_left[0]+j0,
                          grid_index_left[1]+j1,
                          grid_index_left[2]+j2]
    if dims == 4:
        for j0, bf0 in enumerate(BF[0]):
            for j1, bf1 in enumerate(BF[1]):
                for j2, bf2 in enumerate(BF[2]):
                    for j3, bf3 in enumerate(BF[3]):
                        f_interpolated2[i] += bf0*bf1*bf2*bf3 * \
                            f[grid_index_left[0]+j0,
                              grid_index_left[1]+j1,
                              grid_index_left[2]+j2,
                              grid_index_left[3]+j3]
    if dims == 5:
        for j0, bf0 in enumerate(BF[0]):
            for j1, bf1 in enumerate(BF[1]):
                for j2, bf2 in enumerate(BF[2]):
                    for j3, bf3 in enumerate(BF[3]):
                        for j4, bf4 in enumerate(BF[3]):
                            f_interpolated2[i] += bf0*bf1*bf2*bf3*bf4 * \
                                f[grid_index_left[0]+j0,
                                  grid_index_left[1]+j1,
                                  grid_index_left[2]+j2,
                                  grid_index_left[3]+j3,
                                  grid_index_left[4]+j4]
    # ...

t3 = timeit.default_timer()

print(t1-t0)
print(t3-t2)
f_interpolated.shape = (n1_f, n2_f, n3_f)
f_interpolated2.shape = (n1_f, n2_f, n3_f)
# f_interpolated2 = np.transpose(f_interpolated2, (1,0,2))
# print(f_interpolated2.shape)

fig, ax = plt.subplots()
plt.plot(x1, f[:, 0, 0], "go")
plt.plot(x1_f, f_interpolated[:, 0, 0], "r.")
plt.plot(x1_f, f_interpolated2[:, 0, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x2, f[0, :, 0], "go")
plt.plot(x2_f, f_interpolated[0, :, 0], "r.")
plt.plot(x2_f, f_interpolated2[0, :, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x3, f[0, 0, :], "go")
plt.plot(x3_f, f_interpolated[0, 0, :], "r.")
plt.plot(x3_f, f_interpolated2[0, 0, :], "b.")
plt.show()


# j = 0
# for d0 in range(3):  # 2 points along each dimension
#     for d1 in range(3):
#         for d2 in range(3):
#             i0 = lower_i[d0]
#             i1 = lower_i[d1]
#             i2 = lower_i[d2]
#             p[j] = grid[0][lower_i[0]+d0], grid[1][lower_i[1]+d1], grid[2][lower_i[2]+d2]  # ..., grid[n][dn]
#             j +=1
# for j in range(3**dims):
#     f_interpolated[i] += lagrange(p, j, point)
# for d0 in range(2):  # 2 points along each dimension
#     for d1 in range(2):
#         for d2 in range(2):
#             f_interpolated[i] +=

# for lower_i in
# x = np.linspace(0, 1, 100)
# p = lagrange([0, 0.5, 1.0], [1., 0, 0])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# p = lagrange([0, 0.5, 1.0], [0, 1., 0])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# p = lagrange([0, 0.5, 1.0], [0, 0, 1.])
# plt.plot(x, p[2]*x**2+p[1]*x**1+p[0])

# plt.show()


# rgi(xi_new)
# c =
# ndp = NdPPoly(c, x)


# rbs = RectBivariateSpline(x1, x2, f)
