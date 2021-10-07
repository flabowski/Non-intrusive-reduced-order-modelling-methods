#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:04:11 2021

@author: florianma
"""
import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt
from dolfin import (UnitSquareMesh, FunctionSpace, DOLFIN_EPS, Constant, pi,
                    DirichletBC, Function, TrialFunction, TestFunction,
                    Expression, File, plot, solve, UnitIntervalMesh)
from dolfin import (inner, grad, dx, ds, dot)
from ROM.snapshot_manager import Data
from low_rank_model_construction.basis_function_interpolation import interpolateV
import matplotlib
plt.close("all")
cmap = matplotlib.cm.get_cmap('jet')
plot_width = 16


u_manufactured = "x[0] * mu"
f_rhs = "x[0] * mu"

u_manufactured = "sin(x[0]*2*pi) * mu"
f_rhs = "sin(x[0]*2*pi)*mu * (1+mu*4*pi*pi)"

u_manufactured = "sin(x[0]*2*pi) * mu*mu"
f_rhs = "sin(x[0]*2*pi)*mu*mu * (1+mu*4*pi*pi)"

u_manufactured = "sin(x[0]*2*pi*mu)"
f_rhs = "sin(x[0]*2*pi*mu) * (1+4*mu*mu*mu*pi*pi)"


m, n = 15, 11
X = np.empty((m, n))
Xe = np.empty((m, n))
x = [np.linspace(0, 1, m)]
mus = np.linspace(0, 1, n)
# Create mesh and define function space
mesh = UnitIntervalMesh(m - 1)
V = FunctionSpace(mesh, "Lagrange", 1)


def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS


# Define boundary condition
_mu_ = Constant(0.0)
u0 = Expression(u_manufactured, degree=1, mu=_mu_)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
# u1 = Function(V)
# u1.vector().vec().array = 0.
u = TrialFunction(V)
v = TestFunction(V)
f = Expression(f_rhs, degree=1, mu=_mu_)
g = Expression(u_manufactured, degree=1, mu=_mu_)

dt = Constant(1.0)

a = dot(u / dt, v) * dx + _mu_ * inner(grad(u), grad(v)) * dx
L = f * v * dx  # + g*v*ds  # - u1/dt*v*dx

for j, mu in enumerate(mus):
    _mu_.assign(mu)
    u = Function(V)
    solve(a == L, u, bc)
    X[:, j] = u.compute_vertex_values(mesh)  # u.vector().vec().array.ravel()
    Xe[:, j] = eval(u_manufactured)
    # plot(u)


grid = [mus]
my_data = Data(X, grid)
my_data.decompose()
# X_ = my_data.from_reduced_space(my_data.to_reduced_space(X))
# error = my_data.std_rb(X)
singular_values = my_data.S
right_singular_values = my_data.VT
print(X.shape, right_singular_values.shape)
xi = np.linspace(0, 1, 101).reshape(-1, 1)
VT_approx_fine = interpolateV(grid, right_singular_values, xi)
X_approx_fine = my_data.from_reduced_space(VT_approx_fine)
Xe_fine = np.empty_like(X_approx_fine)
for j, mu in enumerate(xi.T[0]):
    Xe_fine[:, j] = eval(u_manufactured)
VT_e_fine = my_data.to_reduced_space(Xe_fine)
r = len(right_singular_values)


# plotting # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# solutions in separate plots
for j, mu in enumerate(mus):
    fig, ax = plt.subplots()
    ax.plot(x[0], X[:, j], "r.", label="FEM solution")
    ax.plot(x[0], Xe[:, j], "g.", label="exact solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("solution for u + {:.1f}* d²u/dx² = {}".format(mu, f_rhs))
    plt.legend()
    plt.show()

fig, ax = plt.subplots()
plt.plot(mus, 2 * X.max(axis=0))
plt.xlabel("mu")
plt.ylabel("amplitude")
plt.show()


# # the reduced order model based on 10 simulations (o) can predict solutions for any mu
# fig, ax = plt.subplots(figsize=(plot_width / 2.54, plot_width / 2.54))
# nrows, ncols = X_approx_fine.shape
# w, h = (mus.max() - mus.min()) / ncols, 1.0 / nrows
# left, right, bottom, top = mus.min() - w / 2, mus.max() + \
#     w / 2, 0 - h / 2, 1 + h / 2
# im2 = plt.imshow(X_approx_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
#                  interpolation='nearest', extent=(left, right, bottom, top))
# for i in range(len(mus)):
#     color = X[:, i]
#     plt.scatter(np.ones_like(x[0]) * mus[i], x[0], s=10, c=color, cmap=cmap,
#                 vmin=X.min(), vmax=X.max(), edgecolor="k")
# plt.colorbar(im2, label="u(x)")
# plt.xlabel("mu")
# plt.ylabel("u(x)")
# plt.title("The ROM based on 10 simulations (o) can predict solutions for any mu")
# plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = X_approx_fine.shape
w, h = (mus.max() - mus.min())/ncols, 1.0/nrows
left, right, bottom, top = mus.min()-w/2, mus.max()+w/2, 0-h/2, 1+h/2
im2 = ax1.imshow(X_approx_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
for i in range(len(mus)):
    color = X[:, i]
    ax1.scatter(np.ones_like(x[0]) * mus[i], x[0], s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")

im2 = ax2.imshow(Xe_fine-X_approx_fine, cmap=cmap, origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im2 = ax3.imshow(Xe_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_xlabel("mu")
ax2.set_xlabel("mu")
ax3.set_xlabel("mu")
ax1.set_ylabel("x")
ax2.set_ylabel("x")
ax3.set_ylabel("x")
plt.suptitle("The ROM based on 10 FEM simulations (o) can approximate solutions for any mu")
ax1.set_title("Approximations based on FEM solutions (o)")
ax2.set_title("error")
ax3.set_title("exact solution: u(x)="+u_manufactured)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = VT_approx_fine.shape
w, h = (mus.max() - mus.min()) / ncols, 1.0
left, right, bottom, top = mus.min()-w/2, mus.max()+w/2, 0+h/2, r+h/2
im2 = ax1.imshow(VT_approx_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
for i in range(len(mus)):
    color = right_singular_values[:, i]
    ax1.scatter(np.ones_like(color) * mus[i], np.arange(1, len(color)+1), s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")
for i in range(r):
    ax1.text(1.02, i+1, "{:.0f} %".format(singular_values[i]/singular_values.sum()*100))

im2 = ax2.imshow(VT_e_fine-VT_approx_fine, cmap=cmap, origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im2 = ax3.imshow(VT_e_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_xlabel("mu")
ax2.set_xlabel("mu")
ax3.set_xlabel("mu")
ax1.set_ylabel("mode # \n(contribution in %)")
ax2.set_ylabel("mode #")
ax3.set_ylabel("mode #")
plt.suptitle("Linear interpolation of the right singular values")
ax1.set_title("Approximations based on FEM solutions (o)")
ax2.set_title("error")
ax3.set_title("exact solutions")
plt.show()
