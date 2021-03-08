#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:28:31 2021

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import matplotlib as mpl
from dolfin import (Function, DirichletBC, Expression, TestFunction, ds,
                    TrialFunction, Mesh, MeshEditor, AutoSubDomain,
                    MeshFunction, FunctionSpace, Constant, VectorFunctionSpace)


class CavityMesh():
    def __init__(self, L, lcar):
        mesh_pygmsh = self.get_pygmsh_mesh(L, lcar)
        self.mesh = self.gmsh2dolfin_2Dmesh(mesh_pygmsh, 0)
        self.points = self.mesh.coordinates()
        self.simplices = self.mesh.cells()
        return

    def get_pygmsh_mesh(self, L, lcar):
        with pygmsh.geo.Geometry() as geom:
            p = [geom.add_point([.0, .0], lcar),
                 geom.add_point([L, .0], lcar),
                 geom.add_point([L, L], lcar),
                 geom.add_point([.0, L], lcar)]
            c = [geom.add_line(p[0], p[1]),
                 geom.add_line(p[1], p[2]),
                 geom.add_line(p[2], p[3]),
                 geom.add_line(p[3], p[0])]
            ll1 = geom.add_curve_loop([c[0], c[1], c[2], c[3]])
            s = [geom.add_plane_surface(ll1)]
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
        cells = msh.cells_dict["triangle"].astype(np.uintp)-unused_points
        mesh = Mesh()
        editor = MeshEditor()
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


class CavityDomain():
    def __init__(self, parameters, mesh):
        """Function spaces and BCs"""
        T_init = Constant(parameters["initial temperature [°C]"])
        self.t_amb = Constant(parameters["ambient temperature [°C]"])
        self.t_feeder = Constant(parameters["temperature feeder [°C]"])
        self.k_top = Constant(parameters["thermal conductivity top [W/(m K)]"])
        self.k_lft = Constant(parameters["thermal conductivity left [W/(m K)]"])
        self.k_btm = Constant(parameters["thermal conductivity bottom [W/(m K)]"])
        self.k_rgt = Constant(parameters["thermal conductivity right [W/(m K)]"])
        U_m = Constant(parameters["mean velocity lid [m/s]"])
        g = self.g = Constant(parameters["gravity [m/s²]"])
        self.dt = Constant(parameters["dt [s]"])
        self.D = Constant(parameters["Diffusivity [-]"])

        self.mesh = mesh
        V = VectorFunctionSpace(mesh, 'P', 2)
        Q = FunctionSpace(mesh, 'P', 1)
        T = FunctionSpace(mesh, 'P', 1)

        ASD1 = AutoSubDomain(top)
        ASD2 = AutoSubDomain(left)
        ASD3 = AutoSubDomain(bottom)
        ASD4 = AutoSubDomain(right)
        mf = MeshFunction("size_t", mesh, 1)
        mf.set_all(9999)
        ASD1.mark(mf, 1)
        ASD2.mark(mf, 2)
        ASD3.mark(mf, 3)
        ASD4.mark(mf, 4)
        self.ds_ = ds(subdomain_data=mf, domain=mesh)

        self.vu, self.vp, self.vt = (TestFunction(V), TestFunction(Q),
                                     TestFunction(T))
        self.u_, self.p_, self.t_ = Function(V), Function(Q), Function(T)
        self.mu, self.rho = Function(T), Function(T)
        # self.mu_k, self.rho_k = Function(T), Function(T)
        self.u_1, self.p_1, self.t_1, self.rho_1 = (Function(V), Function(Q),
                                                    Function(T), Function(T))
        self.u, self.p, self.t = (TrialFunction(V), TrialFunction(Q),
                                  TrialFunction(T))

        # boundary conditions
        no_slip = Constant((0., 0))
        topflow = Expression(("-x[0] * (x[0] - 1.0) * 6.0 * m", "0.0"),
                             m=U_m, degree=2)
        bc0 = DirichletBC(V, topflow, top)
        bc1 = DirichletBC(V, no_slip, left)
        bc2 = DirichletBC(V, no_slip, bottom)
        bc3 = DirichletBC(V, no_slip, right)
        # bc4 = df.DirichletBC(Q, df.Constant(0), top)
        # bc3 = df.DirichletBC(T, df.Constant(800), top)
        self.bcu = [bc0, bc1, bc2, bc3]
        # no boundary conditions for the pressure
        self.bcp = []
        # bcp = [DirichletBC(Q, Constant(750), top)]
        # self.bct = []
        self.bct = [DirichletBC(T, Constant(750), top)]

        self.robin_boundary_terms = (
            self.k_top*(self.t - self.t_feeder)*self.vt*self.ds_(1)
            + self.k_lft*(self.t - self.t_amb)*self.vt*self.ds_(2)
            + self.k_btm*(self.t - self.t_amb)*self.vt*self.ds_(3)
            + self.k_rgt*(self.t - self.t_amb)*self.vt*self.ds_(4))

        # set initial values
        # TODO: find a better solution
        x, y = T.tabulate_dof_coordinates().T
        self.u_1.vector().vec().array[:] = 1e-6
        # self.u_k.vector().vec().array[:] = 1e-6
        self.p_.vector().vec().array[:] = -self.rho.vector().vec().array*g*y
        self.p_1.vector().vec().array[:] = -self.rho.vector().vec().array*g*y
        self.t_1.vector().vec().array = T_init
        self.t_.assign(self.t_1)
        return

    # def rho(self, T):
    #     raise NotImplementedError(self.__class__.__name__ + '.try_something')

    # def mu(self, T, mu_solid):
    #     raise NotImplementedError(self.__class__.__name__ + '.try_something')

    def plot(self):
        cmap = mpl.cm.inferno
        cmap_r = mpl.cm.inferno_r
        u, p, t, m, r = self.u_, self.p_, self.t_, self.mu, self.rho
        mesh = self.mesh
        w0 = u.compute_vertex_values(mesh)
        w0.shape = (2, -1)
        magnitude = np.linalg.norm(w0, axis=0)
        x, y = np.split(mesh.coordinates(), 2, 1)
        u, v = np.split(w0, 2, 0)
        x, y, u, v = x.ravel(), y.ravel(), u.ravel(), v.ravel()
        tri = mesh.cells()
        pressure = p.compute_vertex_values(mesh)
        temperature = t.compute_vertex_values(mesh)
        viscosity = m.compute_vertex_values(mesh)
        density = r.compute_vertex_values(mesh)

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(243, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(244, sharex=ax1, sharey=ax1)
        ax4 = plt.subplot(247, sharex=ax1, sharey=ax1)
        ax5 = plt.subplot(248, sharex=ax1, sharey=ax1)
        ax1.quiver(x, y, u, v, magnitude)
        c2 = ax2.tricontourf(x, y, tri, pressure, levels=40, cmap=cmap)
        c3 = ax3.tricontourf(x, y, tri, temperature, levels=40,
                             vmin=600., vmax=800., cmap=cmap)
        c4 = ax4.tricontourf(x, y, tri, viscosity, levels=40,
                             # vmin=self.mu(800, .1), vmax=self.mu(600, .1),
                             cmap=cmap_r)
        c5 = ax5.tricontourf(x, y, tri, density, levels=40,
                             # vmin=self.rho(800.), vmax=self.rho(600.),
                             cmap=cmap_r)
        plt.colorbar(c2, ax=ax2)
        plt.colorbar(c3, ax=ax3,
                     ticks=[temperature.min(), temperature.max()])
        plt.colorbar(c4, ax=ax4,
                     ticks=[viscosity.min(), viscosity.max()])
        plt.colorbar(c5, ax=ax5,
                     ticks=[density.min(), density.max()])
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax3.set_aspect("equal")
        ax4.set_aspect("equal")
        ax5.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        ax3.set_title("temperature")
        ax4.set_title("viscosity")
        ax5.set_title("density")
        ax1.set_xlim([-.1, 1.1])
        ax1.set_ylim([-.1, 1.1])
        # plt.tight_layout()
        return fig, (ax1, ax2)

    # TODO: throws: "Expecting a function (not <class 'method'>)"
    def top(x, on_boundary):
        return (abs(x[1]-1.0) < 1e-6) & on_boundary


def top(x, on_boundary):
    return (abs(x[1]-1.0) < 1e-6) & on_boundary


def left(x, on_boundary):
    return (x[0] < 1e-6) and on_boundary


def right(x, on_boundary):
    return (abs(x[0]-1.0) < 1e-6) and on_boundary


def walls(x, on_boundary):
    return (abs((x[0]-1.0)*x[0]) < 1e-6) & on_boundary


def bottom(x, on_boundary):
    return (abs(x[1]) < 1e-6) & on_boundary


if __name__ == "__main__":
    pass
