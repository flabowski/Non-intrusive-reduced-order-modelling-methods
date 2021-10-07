# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:32:44 2021

@author: florianma
"""
from scipy.interpolate import (RegularGridInterpolator, RectBivariateSpline,
                               interpn, griddata, Rbf, interp1d)  # RBFInterpolator
import numpy as np
from ROM.snapshot_manager import load_snapshots_cavity  # , plot_snapshot_cav
from ROM.snapshot_manager import Data
# from low_rank_model_construction.proper_orthogonal_decomposition import Data
# import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
plot_width = 16
LINALG_LIB = "numpy"
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


class RightSingularValueInterpolator():
    def __init__(self, points, values, bounds_error=True, fill_value=np.nan):
        pass


def interpolateV(grid, values, xi):
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be uneven.
    The interpolation is repeated r times.

    Parameters
    ----------
    points : n-D ndarray of floats with shapes (m0, m1, ..., mn)
        The points defining the regular grid in n dimensions.
    values : array_like, shape (r, m0, ..., mn, ...)
        The data on the regular grid in n dimensions.
    xi : 2-D ndarray of floats with shape (m, n).
        Points at which to interpolate data.

    Returns
    -------
    V_interpolated : ndarray
        Array of interpolated values.
    """
    #
    # values may be the snapshot matrix shaped m, m0, m1, .. mn
    # or VT shaped r, m0, m1, ..., mn
    # m0: number of timesteps
    # m1: number of different wall temperatures

    # need grid as input: RegularGridInterpolator, BasisFunctionRegularGridInterpolator
    # need points as input: Rbf, CloughTocher2DInterpolator
    # need to reinitialize every time: CloughTocher2DInterpolator, splines
    m, n = xi.shape
    r = values.shape[0]
    for dim in range(n):
        mn = values.shape[dim+1]
        msg = "Mismatching arrays! Got {:.0f} grid points along axis {:.0f} but {:.0f} values.".format(
            len(grid[dim]), dim, mn)
        assert len(grid[dim]) == mn, msg

    XNs = np.meshgrid(*grid, indexing="ij")
    points = np.array([XN.ravel() for XN in XNs]).T
    print(points.shape, values.shape[1:])
    print("interpolating {:.0f} time(s)".format(r))
    # if n == 1:
    #     grid = [points[:].copy()]
    # elif n == 2:
    #     grid = [points[0, :].copy(), points[:, 0].copy()]
    # elif n == 3:
    #     grid = [points[0, :, :].copy(),
    #             points[:, 0, :].copy(),
    #             points[:, :, 0].copy()]
    # elif n == 4:
    #     grid = [points[0, :, :, :].copy(),
    #             points[:, 0, :, :].copy(),
    #             points[:, :, 0, :].copy(),
    #             points[:, :, :, 0].copy()]

    V_interpolated = np.zeros((r, m))
    my_interpolating_function = RegularGridInterpolator(grid, values[0, :])
    my_interpolating_function = Rbf(points, values[0, :])
    for i in range(r):
        # points.shape (400, 2) | vals.shape (400, ) | xi.shape (50, 2)
        vals = values[i, :]
        # vals.shape = n,
        my_interpolating_function.values = vals  # cheaper than initializing new
        my_interpolating_function.di = vals  # cheaper than initializing new
        V_interpolated[i, :] = my_interpolating_function(xi).ravel()
    return V_interpolated
    # V_interpolated[:, i] = griddata(points, vals, xi, method='linear').copy()


class BasisFunctionRegularGridInterpolator(RegularGridInterpolator):
    """
    Interpolation on a regular grid in arbitrary dimensions (c.f. scipy, RegularGridInterpolator)

    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear and nearest-neighbor interpolation are supported. Currently,
    only quadratic basis functions are supported

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    If any of `points` have a dimension of size 1, linear interpolation will
    return an array of `nan` values. Nearest-neighbor interpolation will work
    as usual in this case.
    """

    def __init__(self, points, values, method="basis functions",
                 bounds_error=True, fill_value=np.nan):
        # pukes on unknown method
        RegularGridInterpolator.__init__(
            self, points, values, "linear", bounds_error, fill_value)
        self.method = method

        ndim = len(self.grid)
        if method == "basis functions":
            self.Phi = [None for d in range(ndim)]
            self.ind_grd = [None for d in range(ndim)]
            for d in range(ndim):
                self.Phi[d], self.ind_grd[d] = self._BF_coeff_along(
                    self.grid[d])
        if method == "interpolation functions":
            self._interpolation_functions()

    def _interpolation_functions(self):
        grid = self.grid
        ndim = len(grid)
        values = self.values
        if ndim == 1:
            self.f_x0 = np.array(interp1d(self.grid[0], values),)
        if ndim == 2:
            x0, x1 = grid[0], grid[1]
            n0, n1 = len(grid[0]), len(grid[1])
            self.f_x0 = np.empty((n1,), dtype=object)
            self.f_x1 = np.empty((n0,), dtype=object)
            for j in range(n1):  # functions ALONG x1 (rows)
                print(x0.shape, values[:, j].shape, values.shape)
                self.f_x0[j] = interp1d(x0, values[:, j])
            for i in range(n0):  # functions ALONG columns
                self.f_x1[i] = interp1d(x1, values[i, :])
        if ndim == 3:
            x0, x1, x2 = grid[0], grid[1], grid[2]
            n0, n1, n2 = len(x0), len(x1), len(x2)
            self.f_x0 = np.empty((n1, n2), dtype=object)
            self.f_x1 = np.empty((n0, n2), dtype=object)
            self.f_x2 = np.empty((n0, n1), dtype=object)
            # ALONG x0
            for j in range(n1):
                for k in range(n2):
                    self.f_x0[j, k] = interp1d(x0, values[:, j, k])
            # ALONG x1
            for i in range(n0):
                for k in range(n2):
                    self.f_x1[i, k] = interp1d(x1, values[i, :, k])
            # ALONG x2
            for i in range(n0):
                for j in range(n1):
                    self.f_x2[i, j] = interp1d(x2, values[i, j, :])
        if ndim == 4:
            x0, x1, x2, x3 = grid[0], grid[1], grid[2], grid[3]
            n0, n1, n2, n3 = len(x0), len(x1), len(x2), len(x3)
            self.f_x0 = np.empty((n1, n2, n3), dtype=object)
            self.f_x1 = np.empty((n0, n2, n3), dtype=object)
            self.f_x2 = np.empty((n0, n1, n3), dtype=object)
            self.f_x3 = np.empty((n0, n1, n2), dtype=object)
            # ALONG x0
            for j in range(n1):
                for k in range(n2):
                    for r in range(n3):
                        self.f_x0[j, k, r] = interp1d(x0, values[:, j, k, r])
            # ALONG x1
            for i in range(n0):
                for k in range(n2):
                    for r in range(n3):
                        self.f_x1[i, k, r] = interp1d(x1, values[i, :, k, r])
            # ALONG x2
            for i in range(n0):
                for j in range(n1):
                    for r in range(n3):
                        self.f_x2[i, j, k] = interp1d(x2, values[i, j, :, r])
            # ALONG x3
            for i in range(n0):
                for j in range(n1):
                    for k in range(n2):
                        self.f_x3[i, j, k] = interp1d(x2, values[i, j, k, :])
        if ndim == 5:
            x0, x1, x2, x3, x4 = grid[0], grid[1], grid[2], grid[3], grid[4]
            n0, n1, n2, n3, n4 = len(x0), len(x1), len(x2), len(x3), len(x4)
            self.f_x0 = np.empty((n1, n2, n3, n4), dtype=object)
            self.f_x1 = np.empty((n0, n2, n3, n4), dtype=object)
            self.f_x2 = np.empty((n0, n1, n3, n4), dtype=object)
            self.f_x3 = np.empty((n0, n1, n2, n4), dtype=object)
            self.f_x4 = np.empty((n0, n1, n2, n3), dtype=object)
            # ALONG x0
            for j in range(n1):
                for k in range(n2):
                    for r in range(n3):
                        for s in range(n4):
                            self.f_x0[j, k, r, s] = interp1d(
                                x0, values[:, j, k, r])
            # ALONG x1
            for i in range(n0):
                for k in range(n2):
                    for r in range(n3):
                        for s in range(n4):
                            self.f_x1[i, k, r, s] = interp1d(
                                x1, values[i, :, k, r, s])
            # ALONG x2
            for i in range(n0):
                for j in range(n1):
                    for r in range(n3):
                        for s in range(n4):
                            self.f_x2[i, j, r, s] = interp1d(
                                x2, values[i, j, :, r, s])
            # ALONG x3
            for i in range(n0):
                for j in range(n1):
                    for k in range(n2):
                        for s in range(n4):
                            self.f_x3[i, j, k, s] = interp1d(
                                x2, values[i, j, k, :, s])
            # ALONG x4
            for i in range(n0):
                for j in range(n1):
                    for k in range(n2):
                        for r in range(n3):
                            self.f_x4[i, j, k, r] = interp1d(
                                x2, values[i, j, k, r, :])

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
        # if method not in ["quadratic"]:
        #     raise ValueError("Method '%s' is not defined" % method)
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
                    raise ValueError("One of the requested xi is out of bounds"
                                     " in dimension %d" % i)
        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances,
                                            out_of_bounds)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances,
                                           out_of_bounds)
        if method == "basis functions":
            result = self._evaluate_quadratic(indices, xi, out_of_bounds)
        if method == "interpolation functions":
            result = self._evaluate_interpolation_functions(
                indices, xi, out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_interpolation_functions(self, indices, xi, out_of_bounds):
        grid = self.grid  # 8, 11, 3
        ndim = len(grid)
        for point in xi:
            # print(point)
            if ndim == 3:
                x0, x1, x2 = grid[0], grid[1], grid[2]
                n0, n1, n2 = len(x0), len(x1), len(x2)
                x0, x1, x2 = point
                # ALONG x0
                for j in range(n1):
                    for k in range(n2):
                        self.f_x0[j, k] = interp1d(x0, values[:, j, k])
                # ALONG x1
                for i in range(n0):
                    for k in range(n2):
                        self.f_x1[i, k] = interp1d(x1, values[i, :, k])
                # ALONG x2
                for i in range(n0):
                    for j in range(n1):
                        self.f_x2[i, j] = interp1d(x2, values[i, j, :])

                self.f_x0

                self.f_x1
                self.f_x2

            self.f_x0 = np.empty((n1, n2), dtype=object)
            self.f_x1 = np.empty((n0, n2), dtype=object)
            self.f_x2 = np.empty((n0, n1), dtype=object)

    def _ind_element(self, ind_pt):
        # there are 3 nodes per element, while the side nodes are shared
        return int(ind_pt // 2)

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
        n = (yi - y1) * (yi - y2)
        return (1.0 / n, -(y1 + y2) / n, y1 * y2 / n)

    def _BF_coeff_along(self, xi):
        # basis functions. shaped i, j, d
        # i: global point index (0....n)
        # all points within one rectangle have the same values!
        # j: local point in rectangle index (0, 1, 2)
        # 3: the 3 coefficients (a, b, c) of the polynom a*x**2 + b*x + c
        # note: if the number of points is even, an additional rectangle is added
        # using the points n, n-1, and n-2, even though n-1 and n-2 are used for
        # the second last rectangle too
        # print(xi)
        n_elements = self._ind_element(ind_pt=len(xi))
        Phi_i = np.zeros((n_elements, 3, 3))
        ind_grd = np.zeros((n_elements), dtype=np.int64)
        ind_element = 0
        for i in range(0, len(xi) - 1, 2):
            if (i + 3) > len(xi):
                p = xi[i - 1:i + 2]
                ind_grd[ind_element] = i - 1
            else:
                p = xi[i:i + 3]
                ind_grd[ind_element] = i
            # print(p)
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
                        BF[d][j] = a * point[d]**2 + b * point[d] + c
                    grid_index_left[d] = self.ind_grd[d][i_element]
                if ndim == 1:
                    for j1, bfx in enumerate(BF[0]):
                        result[i] += bfx * self.values[grid_index_left[0] + j1]
                if ndim == 2:
                    for j0, bfx in enumerate(BF[0]):
                        for j1, bfy in enumerate(BF[1]):
                            result[i] += bfx * bfy * \
                                self.values[grid_index_left[0] + j0,
                                            grid_index_left[1] + j1]
                if ndim == 3:
                    for j0, bfx in enumerate(BF[0]):
                        for j1, bfy in enumerate(BF[1]):
                            for j2, bfz in enumerate(BF[2]):
                                result[i] += bfx * bfy * bfz * \
                                    self.values[grid_index_left[0] + j0,
                                                grid_index_left[1] + j1,
                                                grid_index_left[2] + j2]
                if ndim == 4:
                    for j0, bf0 in enumerate(BF[0]):
                        for j1, bf1 in enumerate(BF[1]):
                            for j2, bf2 in enumerate(BF[2]):
                                for j3, bf3 in enumerate(BF[3]):
                                    result[i] += bf0 * bf1 * bf2 * bf3 * \
                                        self.values[grid_index_left[0] + j0,
                                                    grid_index_left[1] + j1,
                                                    grid_index_left[2] + j2,
                                                    grid_index_left[3] + j3]
                if ndim == 5:
                    for j0, bf0 in enumerate(BF[0]):
                        for j1, bf1 in enumerate(BF[1]):
                            for j2, bf2 in enumerate(BF[2]):
                                for j3, bf3 in enumerate(BF[3]):
                                    for j4, bf4 in enumerate(BF[3]):
                                        result[i] += bf0 * bf1 * bf2 * bf3 * bf4 * \
                                            self.values[grid_index_left[0] + j0,
                                                        grid_index_left[1] + j1,
                                                        grid_index_left[2] + j2,
                                                        grid_index_left[3] + j3,
                                                        grid_index_left[4] + j4]
        return result


if __name__ == "__main__":
    path = '/home/florianma@ad.ife.no/Documents/cavity/'

    # [[4, 3015], [5000, 10]]
    # f_name = "50kSVD.npy"
    U = np.load(path + "50k_U.npy")
    S = np.load(path + "50k_S.npy")
    VT = np.load(path + "50k_VT.npy")
    x1 = np.load(path + "Tamb650_time.npy")
    x2 = np.linspace(400, 625, 10)
    # 5240 modes corresponding to a parameterspace shaped (5000, 10)
    VT.shape = (-1, 5000, 10)
    # points.shape = (5000, 10, 2)
    # x1 = points[:, 0, 0]
    # x2 = points[0, :, 1]

    values = VT[0, :, :]
    # interpolateV(points, values, xi)
    n1, n2, n3 = 8, 5, 7
    x1 = np.linspace(0, 8, n1)
    x2 = np.linspace(100, 500, n2)
    x3 = np.linspace(10, 16, n3)
    grid = (x1, x2, x3)
    xx, yy, zz = np.meshgrid(*grid, indexing="ij")
    f = np.random.rand(*xx.shape)
    print(f.shape)

    n = 4
    n1_f = (n1 - 1) * n + 1
    n2_f = (n2 - 1) * n + 1
    n3_f = (n2 - 1) * n + 1
    x1_f = np.linspace(0, 8, n1_f)
    x2_f = np.linspace(100, 500, n2_f)
    x3_f = np.linspace(10, 16, n3_f)
    xx_f, yy_f, zz_f = np.meshgrid(x1_f, x2_f, x3_f, indexing="ij")
    xi_new = np.array([xx_f.ravel(), yy_f.ravel(),
                       zz_f.ravel()]).T  # 3, 495000

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
