#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:51:13 2021

@author: florianma
"""

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pygmsh
<<<<<<< HEAD
from dolfin import VectorElement, FiniteElement, Constant, inner, grad, div, \
    dx, Function, DirichletBC, Expression, solve, lhs, rhs, TestFunction, ds, \
    TrialFunction, dot, nabla_grad, split, errornorm, Mesh, MeshEditor, \
    AutoSubDomain, MeshFunction, FacetNormal, assemble, Identity, \
    project, FunctionSpace, sym, Constant, TestFunctions, VectorFunctionSpace
=======
>>>>>>> 594981552cb3a1ae9adf5a4ec030cc733820fcca


class CylinderMesh():
    def __init__(self, lcar):
        mesh_pygmsh = self.get_pygmsh_mesh(lcar)
        self.mesh = self.gmsh2dolfin_2Dmesh(mesh_pygmsh, 1)
        self.points = self.mesh.coordinates()
        self.simplices = self.mesh.cells()
        return

    def get_pygmsh_mesh(self, lcar):
        with pygmsh.geo.Geometry() as geom:
            r = .05
            p = [geom.add_point([.20, .20], lcar),
                 geom.add_point([0.0, .0], lcar),
                 geom.add_point([2.2, .0], lcar),
                 geom.add_point([2.2, .41], lcar),
                 geom.add_point([0.0, .41], lcar),
                 geom.add_point([.2 + r, .20], lcar),
                 geom.add_point([.20, .2 + r], lcar),
                 geom.add_point([.2 - r, .20], lcar),
                 geom.add_point([.20, .2 - r], lcar)]
            c = [geom.add_line(p[1], p[2]),
                 geom.add_line(p[2], p[3]),
                 geom.add_line(p[3], p[4]),
                 geom.add_line(p[4], p[1]),
                 geom.add_circle_arc(p[5], p[0], p[6]),
                 geom.add_circle_arc(p[6], p[0], p[7]),
                 geom.add_circle_arc(p[7], p[0], p[8]),
                 geom.add_circle_arc(p[8], p[0], p[5])]
            ll1 = geom.add_curve_loop([c[0], c[1], c[2], c[3]])
            ll2 = geom.add_curve_loop([c[4], c[5], c[6], c[7]])
            s = [geom.add_plane_surface(ll1, [ll2])]
            # s = [geom.add_plane_surface(ll1)]
            geom.add_surface_loop(s)
            msh = geom.generate_mesh()
        return msh

    def gmsh2dolfin_2Dmesh(self, msh, unused_points):
        """
        Helping function to create a 2D mesh for FEniCS from a gmsh.
        important! Dont leave any unused points like the center of the circle
        in the node list. FEniCS will crash!
        """
        msh.prune_z_0()
        nodes = msh.points[unused_points:]
        cells = msh.cells_dict["triangle"].astype(np.uintp) - unused_points
        mesh = df.Mesh()
        editor = df.MeshEditor()
        # point, interval, triangle, quadrilateral, hexahedron
        editor.open(mesh, "triangle", 2, 2)
        editor.init_vertices(len(nodes))
        editor.init_cells(len(cells))
        [editor.add_vertex(i, n) for i, n in enumerate(nodes)]
        [editor.add_cell(i, n) for i, n in enumerate(cells)]
        editor.close()
        return mesh

    def plot(self):
        """lets just steal it
        """
        from scipy.spatial import delaunay_plot_2d
        fig = delaunay_plot_2d(self)
        ax = fig.gca()
        ax.set_aspect("equal")
        return fig, ax


class CylinderDomain():
    def __init__(self, parameters, mesh):
        """Function spaces and BCs"""
        V = df.VectorFunctionSpace(mesh, 'P', 2)
        Q = df.FunctionSpace(mesh, 'P', 1)
        self.mesh = mesh
        self.rho = Constant(parameters["density [kg/m3]"])
        self.mu = Constant(parameters["viscosity [Pa*s]"])
        self.dt = Constant(parameters["dt [s]"])
        self.g = 0.0
        self.vu, self.vp = df.TestFunction(V), df.TestFunction(Q)  # for integration
        self.u_, self.p_ = df.Function(V), df.Function(Q)  # for the solution
        self.u_1, self.p_1 = df.Function(V), df.Function(Q)  # for the prev. solution
        self.u_k, self.p_k = df.Function(V), df.Function(Q)  # for the prev. solution
        self.u, self.p = df.TrialFunction(V), df.TrialFunction(Q)  # unknown!

        U_m = parameters["velocity [m/s]"]
        U0_str = "4.*U_m*x[1]*(.41-x[1])/(.41*.41)"
        x = [0, .41 / 2]  # noqa: F841  # evaluate the Expression at the center of the channel
        self.U_mean = np.mean(2 / 3 * eval(U0_str))

        U0 = df.Expression((U0_str, "0"), U_m=U_m, degree=2)
        bc0 = df.DirichletBC(V, df.Constant((0, 0)), cylinderwall)
        bc1 = df.DirichletBC(V, df.Constant((0, 0)), topandbottom)
        bc2 = df.DirichletBC(V, U0, inlet)
        bc3 = df.DirichletBC(Q, df.Constant(0), outlet)
        self.bcu = [bc0, bc1, bc2]
        self.bcp = [bc3]
        # ds is needed to compute drag and lift.
        ASD1 = df.AutoSubDomain(topandbottom)
        ASD2 = df.AutoSubDomain(cylinderwall)
        mf = df.MeshFunction("size_t", mesh, 1)
        mf.set_all(0)
        ASD1.mark(mf, 1)
        ASD2.mark(mf, 2)
        self.ds_ = df.Measure("ds", domain=mesh, subdomain_data=mf)
        return

    def plot(self):
        u, p = self.u_, self.p_
        mesh = self.mesh

        velocity = u.compute_vertex_values(mesh)
        velocity.shape = (2, -1)
        magnitude = np.linalg.norm(velocity, axis=0)
        x, y = mesh.coordinates().T
        u, v = velocity
        tri = mesh.cells()
        pressure = p.compute_vertex_values(mesh)
        # print(x.shape, y.shape, u.shape, v.shape)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, pressure, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)

    # TODO: throws: "Expecting a function (not <class 'method'>)"
    def topandbottom(self, x, on_boundary):
        return (x[1] < 1e-6) or (.4099 < x[1]) and on_boundary

    def cylinderwall(self, x, on_boundary):
        in_circle = ((x[0] - .2) * (x[0] - .2) + (x[1] - .2) * (x[1] - .2)) < 0.0025001
        return (in_circle) & on_boundary

    def inlet(self, x, on_boundary):
        return (x[0] < 1e-6) and on_boundary

    def outlet(self, x, on_boundary):
        return (abs(x[0] - 2.2) < 1e-6) and on_boundary


def topandbottom(x, on_boundary):
    return (x[1] < 1e-6) or (.4099 < x[1]) and on_boundary


def cylinderwall(x, on_boundary):
    in_circle = ((x[0] - .2) * (x[0] - .2) + (x[1] - .2) * (x[1] - .2)) < 0.0025001
    return (in_circle) & on_boundary


def inlet(x, on_boundary):
    return (x[0] < 1e-6) and on_boundary


def outlet(x, on_boundary):
    return (abs(x[0] - 2.2) < 1e-6) and on_boundary


if __name__ == "__main__":
    pass
