# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:32:44 2021

@author: florianma
"""
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf)  # RBFInterpolator
import numpy as np
from nirom.src.cross_validation import load_snapshots_cavity, plot_snapshot_cav
from nirom.low_rank_model_construction.proper_orthogonal_decomposition import Data
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
plot_width = 16
LINALG_LIB = "tensorflow"
timed = True

if LINALG_LIB == "tensorflow":
    qr = tf.linalg.qr
    transpose = tf.transpose
    matmul = tf.matmul
    reshape = tf.reshape
    inv = tf.linalg.inv
elif LINALG_LIB == "numpy":
    qr = np.linalg.qr
    transpose = np.transpose
    matmul = np.matmul
    reshape = np.reshape
    inv = np.linalg.inv

# [[4, 3015], [5000, 10]]
f_name = "50kSVD.npy"
U = np.load("U"+f_name)
S = np.load("S"+f_name)
VT = np.load("VT"+f_name)
points = np.load("xi.npy")
VT.shape = (-1, 5000, 10)
points.shape = (5000, 10, 2)
x1 = points[:, 0, 0]
x2 = points[0, :, 1]

values = VT[0, :, :]


def interpolateV(points, values, xi):
    """
    Parameters
    ----------
    points : 2-D ndarray of floats with shape (m, D), or length D tuple of 1-D ndarrays with shape (m,).
        Data point coordinates.
    values : ndarray of float or complex, shape (m, r). V Matrix from SVD (not V.T!)
        Data values.
    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.

    Returns
    -------
    V_interpolated : ndarray
        Array of interpolated values.

    n: n_modes = n_nodes*4 (u,v,p,t)
    D: 2 (time and Tamb or mu)
    r: 12 reduced rank
    """
    m, D = points.shape
    m, r = values.shape  # m, n_modes
    d1, D = xi.shape  # snapshots_per_dataset
    d2 = m // d1  # n_trainingsets
    assert m == d1*d2, "?"

    V_interpolated = np.zeros((d1, r))
    for i in range(r):
        # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
        vals = values[:, i].numpy().copy()

        if vals.shape[0] != points.shape[0]:
            # print("repeat each oscillation")
            # print(points.shape, vals.shape, xi.shape)
            d_ = vals.reshape(d1, d2)  # 50, 1
            # print(d_.shape)
            # 8, 3*150 repeats each oscillation
            d = np.concatenate((d_, d_, d_), axis=0)
            # print(d.shape)
            vals = d.ravel()
            # print(np.allclose(vals, d.ravel()))
            # print(2, points.shape, vals.shape, xi.shape)

        # methods:
        # "linear" or "nearest" call RegularGridInterpolator
        #  "splinef2d" calls RectBivariateSpline
        v = interpn(points, vals, xi)  # includes expensive triangulation!

        # methods:
        # 'linear' calls LinearNDInterpolator
        # 'nearest' calls NearestNDInterpolator
        # 'cubic' calls CloughTocher2DInterpolator
        v = griddata(points, vals, xi, method='linear')

        rbfi = Rbf(points[..., 0], points[..., 1], vals)
        v = rbfi(xi[..., 0], xi[..., 1])

        V_interpolated[:, i] = v.copy()
    return V_interpolated
    # V_interpolated[:, i] = griddata(points, vals, xi, method='linear').copy()


class BasisFunctionRegularGridInterpolator(RegularGridInterpolator):
    def __init__(self, points, values, bounds_error=True, fill_value=np.nan):
        RegularGridInterpolator.__init__(self, points, values, method="linear",
                                         bounds_error=bounds_error, fill_value=fill_value)

        ndim = len(self.grid)
        self.Phi = [None for d in range(ndim)]
        self.ind_grd = [None for d in range(ndim)]
        for d in range(ndim):
            self.Phi[d], self.ind_grd[d] = self._BF_coeff_along(self.grid[d])
        # if method not in ["linear", "nearest"]:
        #     raise ValueError("Method '%s' is not defined" % method)
        # self.method = method
        # self.bounds_error = bounds_error

        # if not hasattr(values, 'ndim'):
        #     # allow reasonable duck-typed values
        #     values = np.asarray(values)

        # if len(points) > values.ndim:
        #     raise ValueError("There are %d point arrays, but values has %d "
        #                      "dimensions" % (len(points), values.ndim))

        # if hasattr(values, 'dtype') and hasattr(values, 'astype'):
        #     if not np.issubdtype(values.dtype, np.inexact):
        #         values = values.astype(float)

        # self.fill_value = fill_value
        # if fill_value is not None:
        #     fill_value_dtype = np.asarray(fill_value).dtype
        #     if (hasattr(values, 'dtype') and not
        #             np.can_cast(fill_value_dtype, values.dtype,
        #                         casting='same_kind')):
        #         raise ValueError("fill_value must be either 'None' or "
        #                          "of a type compatible with values")

        # for i, p in enumerate(points):
        #     print(i, p.shape, values.shape)
        #     if not np.all(np.diff(p) > 0.):
        #         raise ValueError("The points in dimension %d must be strictly "
        #                          "ascending" % i)
        #     if not np.asarray(p).ndim == 1:
        #         raise ValueError("The points in dimension %d must be "
        #                          "1-dimensional" % i)
        #     if not values.shape[i] == len(p):
        #         raise ValueError("There are %d points and %d values in "
        #                          "dimension %d" % (len(p), values.shape[i], i))
        # self.grid = tuple([np.asarray(p) for p in points])
        # self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported is "quadratic"
            "linear", "cubic" and "nearest".

        """
        method = self.method if method is None else method
        if method not in ["quadratic"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        # xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        result = self._evaluate_quadratic(indices, xi, out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _ind_element(self, ind_pt):
        # there are 3 nodes per element, while the side nodes are shared
        return int(ind_pt//2)

    def quadratic_coeff(self, p, i):
        assert len(p) == 3, "need 3 points for quadratic basis function"
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

    def _BF_coeff_along(self, xi):
        # basis functions. shaped i, j, d
        # i: global point index (0....n)
        # all points within one rectangle have the same values!
        # j: local point in rectangle index (0, 1, 2)
        # 3: the 3 coefficients (a, b, c) of the polynom a*x**2 + b*x + c
        # note: if the number of points is even, an additional rectangle is added
        # using the points n, n-1, and n-2, even though n-1 and n-2 are used for
        # the second last rectangle too
        print(xi)
        n_elements = self._ind_element(ind_pt=len(xi))
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
            Phi_i[ind_element, 0] = self.quadratic_coeff(p, 0)
            Phi_i[ind_element, 1] = self.quadratic_coeff(p, 1)
            Phi_i[ind_element, 2] = self.quadratic_coeff(p, 2)
            ind_element += 1
        return Phi_i, ind_grd

    def _evaluate_quadratic(self, indices, xi_new, out_of_bounds):
        ndim = len(self.grid)
        # TODO: enable broadcasting for f (or just call this multiple times?)
        result = np.zeros(len(xi_new))
        for i, point in enumerate(xi_new):
            if not out_of_bounds[i]:
                grid_index_left = [None for d in range(ndim)]
                BF = [[None for j in range(3)] for d in range(ndim)]
                d = 0
                for d in range(ndim):
                    # BF[d] is lpx, lpy, lpz
                    # Phi[d] is Phix, Phiy, Phiz
                    # point[d] is x, y, z
                    i_element = self._ind_element(indices[d][i])
                    for j in range(3):
                        a, b, c = self.Phi[d][i_element, j, :]
                        BF[d][j] = a*point[d]**2 + b*point[d] + c
                    grid_index_left[d] = self.ind_grd[d][i_element]
                if ndim == 1:
                    for j1, bfx in enumerate(BF[0]):
                        result[i] += bfx * self.values[grid_index_left[0]+j1]
                if ndim == 2:
                    for j0, bfx in enumerate(BF[0]):
                        for j1, bfy in enumerate(BF[1]):
                            result[i] += bfx*bfy * \
                                self.values[grid_index_left[0]+j0,
                                            grid_index_left[1]+j1]
                if ndim == 3:
                    for j0, bfx in enumerate(BF[0]):
                        for j1, bfy in enumerate(BF[1]):
                            for j2, bfz in enumerate(BF[2]):
                                result[i] += bfx*bfy*bfz * \
                                    self.values[grid_index_left[0]+j0,
                                                grid_index_left[1]+j1,
                                                grid_index_left[2]+j2]
                if ndim == 4:
                    for j0, bf0 in enumerate(BF[0]):
                        for j1, bf1 in enumerate(BF[1]):
                            for j2, bf2 in enumerate(BF[2]):
                                for j3, bf3 in enumerate(BF[3]):
                                    result[i] += bf0*bf1*bf2*bf3 * \
                                        self.values[grid_index_left[0]+j0,
                                                    grid_index_left[1]+j1,
                                                    grid_index_left[2]+j2,
                                                    grid_index_left[3]+j3]
                if ndim == 5:
                    for j0, bf0 in enumerate(BF[0]):
                        for j1, bf1 in enumerate(BF[1]):
                            for j2, bf2 in enumerate(BF[2]):
                                for j3, bf3 in enumerate(BF[3]):
                                    for j4, bf4 in enumerate(BF[3]):
                                        result[i] += bf0*bf1*bf2*bf3*bf4 * \
                                            self.values[grid_index_left[0]+j0,
                                                        grid_index_left[1]+j1,
                                                        grid_index_left[2]+j2,
                                                        grid_index_left[3]+j3,
                                                        grid_index_left[4]+j4]
        return result


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

rgi = BasisFunctionRegularGridInterpolator((x1, x2, x3), values=f)

f_interpolated = rgi(xi_new)
f_interpolated.shape = (n1_f, n2_f, n3_f)
# f_interpolated2 = np.transpose(f_interpolated2, (1,0,2))
# print(f_interpolated2.shape)

fig, ax = plt.subplots()
plt.plot(x1, f[:, 0, 0], "go")
plt.plot(x1_f, f_interpolated[:, 0, 0], "r.")
# plt.plot(x1_f, f_interpolated2[:, 0, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x2, f[0, :, 0], "go")
plt.plot(x2_f, f_interpolated[0, :, 0], "r.")
# plt.plot(x2_f, f_interpolated2[0, :, 0], "b.")
plt.show()

fig, ax = plt.subplots()
plt.plot(x3, f[0, 0, :], "go")
plt.plot(x3_f, f_interpolated[0, 0, :], "r.")
# plt.plot(x3_f, f_interpolated2[0, 0, :], "b.")
plt.show()
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
# https://docs.scipy.org/doc/scipy/reference/interpolate.html

# For data on a grid:

# interpn(points, values, xi[, method, …])
# Multidimensional interpolation on regular grids.

# RegularGridInterpolator(points, values[, …])
# Interpolation on a regular grid in arbitrary dimensions

# RectBivariateSpline(x, y, z[, bbox, kx, ky, s])
# Bivariate spline approximation over a rectangular mesh.

# See also
# scipy.ndimage.map_coordinates

# Tensor product polynomials:
# NdPPoly(c, x[, extrapolate])
# Piecewise tensor product polynomial


# CloughTocher2DInterpolator
# bisplrep(s=0)
# rgi = RegularGridInterpolator((x1, x2), f, bounds_error=False)
# indices, norm_distances, out_of_bounds = rgi._find_indices(xi_new.T)
# rgi(xi_new)
# rbs = RectBivariateSpline(x1, x2, f)
# griddata  # unstructured

# The BivariateSpline class is the 2-D analog of the UnivariateSpline class


# VT_ = matmul(transpose(U)*S[:, None], X)
# if "my_data" not in locals():
#     path = "C:/Users/florianma/Documents/data/freezing_cavity/"
#     # X_all, _xi_all_, x, y, tri, dims_all, phase_length = load_snapshots_cavity(path)
#     my_data = Data(*load_snapshots_cavity(path))

# X_train, X_valid, xi_train, xi_valid = my_data.split(3)
# VT_train = matmul(transpose(U/S), X_train)


# find unique x, y, z, ...


# x = np.linspace(0, 2000, 100)
# i = 1


# # u(x) = sum(ci, Psi_i)
# u = np.array([1900, 1.2])

# zeta = (x-x[i]) / (x[i+1]-x[i])  # in [0, 1]

# # follows from u(zeta, eta) = c1 + c2*eta + c3*eta + c4*zeta*eta
# A = np.array([[1,  0, 0,  0],
#               [-1,  1, 0,  0],
#               [-1,  0, 0,  1],
#               [1, -1, 1, -1]])

# c = matmul(A, u)


# def N1(zeta, eta):
#     return (1-zeta)*(1-eta)


# def N2(zeta, eta):
#     return zeta*(1-eta)


# def N3(zeta, eta):
#     return zeta*eta


# def N4(zeta, eta):
#     return (1-zeta)*eta


# def xy(zeta, eta, P1, P2, P3, P4):
#     x_ = np.array([[zeta], [eta], [1.0]])
#     x1, y1 = P1
#     x2, y2 = P2
#     x3, y3 = P3
#     x4, y4 = P4
#     A = np.array([[x2-x1, x4-x1, x1],
#                   [y2-y1, y4-y1, y1],
#                   [0, 0, 1]])
#     return matmul(inv(A), x_)  # [:-1]


# def zeta_eta(x, y, P1, P2, P3, P4):
#     xi = np.array([[x], [y], [1.0]])
#     x1, y1 = P1
#     x2, y2 = P2
#     x3, y3 = P3
#     x4, y4 = P4
#     A = np.array([[x2-x1, x4-x1, x1],
#                   [y2-y1, y4-y1, y1],
#                   [0, 0, 1]])
#     return matmul(A, xi)  # [:-1]


# # class interpolator:

#    # def __init__(self, points, vals)


# def my_griddata(points, vals, xi, method='linear'):
#     xi = my_data.xi
#     xi.shape = (my_data.d1, my_data.d2, -1)
#     assert np.allclose(xi[:, 0, 0], xi[:, -1, 0])
#     assert np.allclose(xi[0, :, 1], xi[-1, :, 1])
#     x1 = xi[:, 0, 0]
#     x2 = xi[0, :, 1]
#     f = VT[0, :].reshape(my_data.d1, my_data.d2)
#     xi_new = np.array([[0.1, 510], [1, 510], [2, 510]])
#     ind_x1 = np.arange(len(x1))
#     ind_x2 = np.arange(len(x2))
#     for k in range(len(xi_new)):
#         # TODO: make sure its in range
#         x, y = xi_new[k]
#         i = ind_x1[x1 < x][-1]
#         j = ind_x2[x2 < y][-1]
#         print(i, j)
#         P1 = x1[i], x2[i]
#         P2 = x1[i+1], x2[i]
#         P3 = x1[i+1], x2[i+1]
#         P4 = x1[i], x2[i+1]
#         z, e = zeta_eta(x, y, P1, P2, P3, P4)
#         f1 = f[i, j]
#         f2 = f[i+1, j]
#         f3 = f[i+1, j+1]
#         f4 = f[i, j+1]


# xi = my_data.xi
# xi.shape = (my_data.d1, my_data.d2, -1)
# assert np.allclose(xi[:, 0, 0], xi[:, -1, 0])
# assert np.allclose(xi[0, :, 1], xi[-1, :, 1])
# x1 = xi[:, 0, 0]
# x2 = xi[0, :, 1]
# f = VT[0, :].reshape(my_data.d1, my_data.d2)
# xi_new = np.array([[0.1, 510], [1, 510], [2, 510]])
# ind_x1 = np.arange(len(x1))
# ind_x2 = np.arange(len(x2))
# rgi = RegularGridInterpolator((x1, x2), f, bounds_error=False)
# indices, norm_distances, out_of_bounds = rgi._find_indices(xi_new.T)
# rgi(xi_new)
# rbs = RectBivariateSpline(x1, x2, f)

# griddata
# cs = CubicSpline(x_b, f)

# rgi._find_indices(np.array([[-10.1,   1.,   2.],
#                             [500., 510., 510.]]))


# def psi(j):
#     # for j in range(len(xi)):
#     psi_j = 1
#     for m in [j-1, j+1]:
#         psi_j *= (x-xi[m, 0]) / (xi[j, 0]-xi[m, 0])
#     return psi_j


# fig, ax = plt.subplots()
# plt.plot(x, psi_j, "r.")

# problem: non constant parameters e.g. energy in the system, average temperature. Does not fit on a regular grid!
# this is a problem for the NIROM in general.
# problem: non constant parameters e.g. energy in the system, average temperature. Does not fit on a regular grid!
# this is a problem for the NIROM in general.
