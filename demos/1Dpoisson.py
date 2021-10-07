"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2012-11-12

# Begin demo
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

# u_manufactured = "sin(x[0]*2*pi) * mu"
# f_rhs = "sin(x[0]*2*pi)*(1+mu*4*pi*pi)*mu "

# u_manufactured = "sin(x[0]*2*pi) * mu*mu"
# f_rhs = "sin(x[0]*2*pi)*(1-mu*pi*pi*2*2)*mu*mu"

# u_manufactured = "sin(x[0]*2*pi*mu)"
# f_rhs = "sin(x[0]*2*pi*mu)*(1-4*mu*mu*mu*pi*pi)"


m, n = 25, 11
X = np.empty((m, n))
Xe = np.empty((m, n))
x = [np.linspace(0, 1, m)]
mus = np.linspace(0, 1, n)
# Create mesh and define function space
mesh = UnitIntervalMesh(m-1)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)


def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS


# Define boundary condition
_mu_ = Constant(0.0)
u0 = Expression(u_manufactured, degree=1, mu=_mu_)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u1 = Function(V)
u1.vector().vec().array = 0.
u = TrialFunction(V)
v = TestFunction(V)
f = -Expression(f_rhs, degree=1, mu=_mu_)
g = Expression(u_manufactured, degree=1, mu=_mu_)

dt = Constant(1.0)

a = dot(u / dt, v)*dx + _mu_*inner(grad(u), grad(v))*dx
L = f*v*dx  # + g*v*ds  # - u1/dt*v*dx

for j, mu in enumerate(mus):
    _mu_.assign(mu)
    u = Function(V)
    solve(a == L, u, bc)
    X[:, j] = u.vector().vec().array.ravel()
    Xe[:, j] = eval(u_manufactured)
    plot(u)


grid = [mus]
my_data = Data(X, grid)
my_data.decompose()
values = my_data.VT
xi = np.linspace(0, 1, 101).reshape(-1, 1)
VT_interpolated = interpolateV(grid, values, xi)
X_approx = my_data.from_reduced_space(VT_interpolated)

# plotting:
for j, mu in enumerate(mus):
    fig, ax = plt.subplots()
    ax.plot(x[0], X[:, j], "bo")
    ax.plot(x[0], Xe[:, j], "go")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

fig, ax = plt.subplots()
plt.plot(mus, 2*X.max(axis=0))
plt.xlabel("mu")
plt.ylabel("amplitude")
plt.show()


# X_approx = np.random.rand(*X_approx.shape)
fig, ax = plt.subplots(figsize=(plot_width/2.54, plot_width/2.54))
nrows, ncols = X_approx.shape
w, h = (mus.max()-mus.min()) / ncols, 1.0 / nrows
left, right, bottom, top = mus.min()-w/2, mus.max()+w/2, 0-h/2, 1+h/2
im2 = plt.imshow(X_approx, cmap=cmap, vmin=X.min(), vmax=X.max(), origin="lower",
                 interpolation='nearest', extent=(left, right, bottom, top))
for i in range(len(mus)):
    color = X[:, i]
    plt.scatter(np.ones_like(x[0])*mus[i], x[0], s=10, c=color, cmap=cmap,
                vmin=X.min(), vmax=X.max(), edgecolor="k")
plt.colorbar(im2, label="u(x)")
plt.xlabel("mu")
plt.ylabel("amplitude")
plt.show()
