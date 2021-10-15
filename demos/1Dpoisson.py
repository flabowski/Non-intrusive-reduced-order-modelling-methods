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
from low_rank_model_construction.basis_function_interpolation import interpolateV, RightSingularValueInterpolator
import matplotlib
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close("all")
cmap = matplotlib.cm.get_cmap('jet')
# cmap = matplotlib.cm.get_cmap('parula')
cmap = matplotlib.cm.get_cmap('viridis')
plot_width = 12


u_manufactured = "x[0] * mu"
f_rhs = "x[0] * mu"

# u_manufactured = "sin(x[0]*2*pi) * mu"
# f_rhs = "sin(x[0]*2*pi)*mu * (1+mu*4*pi*pi)"

# u_manufactured = "sin(x[0]*2*pi) * mu*mu"
# f_rhs = "sin(x[0]*2*pi)*mu*mu * (1+mu*4*pi*pi)"

u_manufactured = "sin(x[0]*2*pi*mu)"
f_rhs = "sin(x[0]*2*pi*mu) * (1+4*mu*mu*mu*pi*pi)"

# u_manufactured = "sin(x[0]*2*pi*mu*3)"
# f_rhs = "sin(x[0]*2*pi*mu*3) * (1+9*4*mu*mu*mu*pi*pi)"


m, n = 15, 11
n2 = 100+1
X = np.empty((m, n))
Xe = np.empty((m, n))
x = [np.linspace(0, 1, m)]
xi = np.linspace(0, 1, n2).reshape(-1, 1)
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

    
def plot_lines(ax, c, p, pf, X_f, Xe_f, X, Xe, i):
    c = new_colors[i % 10]
    lbl1 = lbl2 = lbl3 = lbl4 = None
    if i == 0:
        lbl1 = "approximation"
        lbl2 = "exact solution"
        lbl3 = "FEM solution"
        lbl4 = "exact solution"
    ax.plot(p, X, color=c, marker="s", linestyle="", label=lbl3)
    # ax.plot(p, Xe, "k.", label=lbl4)
    ax.plot(pf, X_f, color=c, marker=".", linestyle="", label=lbl1)
    ax.plot(pf, Xe_f, color=c, marker="", label=lbl2)
    return

grid = [mus]
my_data = Data(X, grid)
my_data.decompose(eps=1.0-1e-4)
print(my_data.S)
# X_ = my_data.from_reduced_space(my_data.to_reduced_space(X))
# error = my_data.std_rb(X)
singular_values = my_data.S
VT = my_data.VT
# VT_approx_fine = interpolateV(grid, VT, xi, method="nearest")
for method in ['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
               'quadratic', 'cubic', 'previous', 'next']:
    intpld = RightSingularValueInterpolator(grid, VT, method=method)
    VT_approx_fine = intpld(xi)
    X_approx_fine = my_data.from_reduced_space(VT_approx_fine)
    
    # X_approx_fine_interpolated_directly = interpolateV(grid, X, xi)
    intpld = RightSingularValueInterpolator(grid, X, method=method)
    X_approx_fine_interpolated_directly = intpld(xi)
    VT_approx_fine_interpolated_directly = my_data.to_reduced_space(X_approx_fine_interpolated_directly)

    Xe_fine = np.empty_like(X_approx_fine)
    for j, mu in enumerate(xi.T[0]):
        Xe_fine[:, j] = eval(u_manufactured)
    VTe_fine = my_data.to_reduced_space(Xe_fine)
    VTe = my_data.to_reduced_space(Xe)
    r = len(VT)
    
    # print(X_approx_fine-Xe_fine)
    error = my_data.std_rb(Xe_fine, X_approx_fine)
    print("{:.12f}".format(error), method, error)


    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(2*plot_width/2.54, plot_width/2.54))
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    print(my_data.std_rb(X_approx_fine_interpolated_directly, X_approx_fine))
    print(my_data.std_rb(VT_approx_fine, VT_approx_fine_interpolated_directly))
    for i in range(m):
        c = new_colors[i % 10]
        plot_lines(ax1, c, mus, xi.T[0], X_approx_fine[i, :], Xe_fine[i, :], X[i, :], Xe[i, :], i)
        ax1.plot(xi.T[0], X_approx_fine_interpolated_directly[i, :], "k.")
        # plot_lines(ax1, c, mus, xi.T[0], X_approx_fine[i, :], Xe_fine[i, :], X[i, :], Xe[i, :], i)
        if i < r:
            plot_lines(ax2, c, mus, xi.T[0], VT_approx_fine[i, :], VTe_fine[i, :], VT[i, :], VTe[i, :], i)
            ax2.plot(xi.T[0], VT_approx_fine_interpolated_directly[i, :], "k.")
    ax1.set_title("physical space (x varies with color)")
    ax2.set_title("reduced space (mode # varies with color)")
    ax1.legend()
    ax2.legend()
    ax1.set_xlim([0, 1])
    ax1.set_xlabel("mu")
    ax2.set_xlabel("mu")
    ax1.set_ylabel("u(x)")
    ax2.set_ylabel("u(x)")
    plt.show()



# plotting # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# solutions in separate plots
for j, mu in enumerate(mus):
    if j == 10:
        fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
        ax.plot(x[0], X[:, j], "r.", label="FEM solution")
        ax.plot(x[0], Xe[:, j], "g.", label="exact solution")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("manufactured solution for u = {}".format(u_manufactured.replace("[0]", "")))
        plt.legend()
        plt.show()

fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
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
im1 = ax1.imshow(X_approx_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
for i in range(len(mus)):
    color = X[:, i]
    ax1.scatter(np.ones_like(x[0]) * mus[i], x[0], s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")

im2 = ax2.imshow(Xe_fine-X_approx_fine, cmap=cmap, origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im3 = ax3.imshow(Xe_fine, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
cax1 = make_axes_locatable(ax1).append_axes('bottom', size='5%', pad=.5)
cax2 = make_axes_locatable(ax2).append_axes('bottom', size='5%', pad=.5)
cax3 = make_axes_locatable(ax3).append_axes('bottom', size='5%', pad=.5)
fig.colorbar(im1, cax=cax1, orientation="horizontal")
fig.colorbar(im2, cax=cax2, orientation="horizontal")
fig.colorbar(im3, cax=cax3, orientation="horizontal")
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
ax1.set_xlabel("mu")
ax2.set_xlabel("mu")
ax3.set_xlabel("mu")
ax1.set_ylabel("x")
ax2.set_ylabel("x")
ax3.set_ylabel("x")
plt.suptitle("The ROM is based on 10 FEM simulations (o) and can approximate solutions for any mu")
ax1.set_title("Approximations based on FEM solutions (o)")
ax2.set_title("error")
ax3.set_title("exact solution: u(x)="+u_manufactured)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(3*plot_width/2.54, plot_width/2.54))
nrows, ncols = VT_approx_fine.shape
w, h = (mus.max() - mus.min()) / ncols, 1.0
left, right, bottom, top = mus.min()-w/2, mus.max()+w/2, 0+h/2, r+h/2
im1 = ax1.imshow(VT_approx_fine, cmap=cmap, vmin=VT.min(), vmax=VT.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
for i in range(len(mus)):
    color = VT[:, i]
    ax1.scatter(np.ones_like(color) * mus[i], np.arange(1, len(color)+1), s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")
for i in range(r):
    ax1.text(1.02, i+1, "{:.0f} %".format(singular_values[i]/singular_values.sum()*100))

im2 = ax2.imshow(VTe_fine-VT_approx_fine, cmap=cmap, origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
im3 = ax3.imshow(VTe_fine, cmap=cmap, vmin=VT.min(), vmax=VT.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
cax1 = make_axes_locatable(ax1).append_axes('bottom', size='5%', pad=.5)
cax2 = make_axes_locatable(ax2).append_axes('bottom', size='5%', pad=.5)
cax3 = make_axes_locatable(ax3).append_axes('bottom', size='5%', pad=.5)
fig.colorbar(im1, cax=cax1, orientation="horizontal")
fig.colorbar(im2, cax=cax2, orientation="horizontal")
fig.colorbar(im3, cax=cax3, orientation="horizontal")
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



fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(2*plot_width/2.54, plot_width/2.54))
new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_lines(ax, c, p, pf, X_f, Xe_f, X, Xe, i):
    c = new_colors[i % 10]
    lbl1 = lbl2 = lbl3 = lbl4 = None
    if i == 0:
        lbl1 = "approximation"
        lbl2 = "exact solution"
        lbl3 = "FEM solution"
        lbl4 = "exact solution"
    ax.plot(p, X, color=c, marker="s", linestyle="", label=lbl3)
    # ax.plot(p, Xe, "k.", label=lbl4)
    ax.plot(pf, X_f, color=c, marker=".", linestyle="", label=lbl1)
    ax.plot(pf, Xe_f, color=c, marker="", label=lbl2)
    return

for i in range(m):
    c = new_colors[i % 10]
    plot_lines(ax1, c, mus, xi.T[0], X_approx_fine[i, :], Xe_fine[i, :], X[i, :], Xe[i, :], i)
    if i < r:
        plot_lines(ax2, c, mus, xi.T[0], VT_approx_fine[i, :], VTe_fine[i, :], VT[i, :], VTe[i, :], i)
ax1.set_title("physical space (x varies with color)")
ax2.set_title("reduced space (mode # varies with color)")
ax1.legend()
ax2.legend()
ax1.set_xlim([0, 1])
ax1.set_xlabel("mu")
ax2.set_xlabel("mu")
ax1.set_ylabel("u(x)")
ax2.set_ylabel("u(x)")
plt.show()
