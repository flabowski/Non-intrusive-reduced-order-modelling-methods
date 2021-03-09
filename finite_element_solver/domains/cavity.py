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
from dolfin import (Function, DirichletBC, Expression, TestFunction,
                    TrialFunction, Mesh, FunctionSpace, Constant, Measure,
                    VectorFunctionSpace, XDMFFile, MeshValueCollection, cpp)
from mpi4py import MPI
import meshio
from finite_element_solver.domains.cylinder import create_entity_mesh
# TODO: make generic module


def create_cavity_mesh(lcar, L=1.0):
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
        fluid = geom.add_surface_loop(s)
        # Add physical markers to gmsh
        # Triangles:
        # - 0: Fluid
        # Lines:
        # - 1: bottom
        # - 2: right
        # - 3: top
        # - 4: left
        btm = [c[0]]
        rgt = [c[1]]
        top = [c[2]]
        lft = [c[3]]
        geom.add_physical(fluid, "0")
        geom.add_physical(btm, "1")
        geom.add_physical(rgt, "2")
        geom.add_physical(top, "3")
        geom.add_physical(lft, "4")
        # When using physical markers, unused points are placed last in the mesh
        msh = geom.generate_mesh(dim=2)

    # Write mesh to XDMF which we can easily read into dolfin
    if MPI.COMM_WORLD.rank == 0:
        #  Pick one of ['abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml',
        # 'exodus', 'flac3d', 'gmsh', 'gmsh22', 'h5m', 'hmf', 'mdpa', 'med',
        # 'medit', 'nastran', 'neuroglancer', 'obj', 'off', 'permas', 'ply',
        # 'stl', 'su2', 'svg', 'tecplot', 'tetgen', 'ugrid', 'vtk', 'vtu',
        # 'wkt', 'xdmf'], xdmf fails
        input_mesh = create_entity_mesh(msh, "triangle", True, True)
        meshio.write("mesh.xdmf", input_mesh, file_format="xdmf")
        # meshio.write("mesh.xdmf", input_mesh)
        meshio.write("mf.xdmf", create_entity_mesh(msh, "line", True),
                     file_format="xdmf")
    MPI.COMM_WORLD.barrier()


class CavityProblemSetup():
    def __init__(self, parameters, mesh_name, facet_name,
                 bc_dict={"bottom": 1, "right": 2, "top": 3, "left": 4}):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        self.mesh = Mesh()
        with XDMFFile(mesh_name) as infile:
            infile.read(self.mesh)

        mvc = MeshValueCollection("size_t", self.mesh,
                                  self.mesh.topology().dim() - 1)
        with XDMFFile(facet_name) as infile:
            infile.read(mvc, "name_to_read")
        mf = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

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

        V = VectorFunctionSpace(self.mesh, 'P', 2)
        Q = FunctionSpace(self.mesh, 'P', 1)
        T = FunctionSpace(self.mesh, 'P', 1)

        self.ds_ = Measure("ds", domain=self.mesh, subdomain_data=mf)
        # print(self.ds_().subdomain_data().array())
        # print(np.unique(self.ds_().subdomain_data().array()))

        self.vu, self.vp, self.vt = (TestFunction(V), TestFunction(Q),
                                     TestFunction(T))
        self.u_, self.p_, self.t_ = Function(V), Function(Q), Function(T)
        self.mu, self.rho = Function(T), Function(T)
        self.u_1, self.p_1, self.t_1, self.rho_1 = (Function(V), Function(Q),
                                                    Function(T), Function(T))
        self.u, self.p, self.t = (TrialFunction(V), TrialFunction(Q),
                                  TrialFunction(T))

        # boundary conditions
        no_slip = Constant((0., 0))
        topflow = Expression(("-x[0] * (x[0] - 1.0) * 6.0 * m", "0.0"),
                             m=U_m, degree=2)

        bc0 = DirichletBC(V, topflow, mf, bc_dict["top"])
        bc1 = DirichletBC(V, no_slip, mf, bc_dict["left"])
        bc2 = DirichletBC(V, no_slip, mf, bc_dict["bottom"])
        bc3 = DirichletBC(V, no_slip, mf, bc_dict["right"])
        # bc4 = df.DirichletBC(Q, df.Constant(0), top)
        # bc3 = df.DirichletBC(T, df.Constant(800), top)
        self.bcu = [bc0, bc1, bc2, bc3]
        # no boundary conditions for the pressure
        self.bcp = []
        # bcp = [DirichletBC(Q, Constant(750), top)]
        self.bct = []
        # self.bct = [DirichletBC(T, Constant(750), mf, bc_dict["top"])]

        self.robin_boundary_terms = (
            self.k_btm*(self.t - self.t_amb)*self.vt*self.ds_(1)
            + self.k_rgt*(self.t - self.t_amb)*self.vt*self.ds_(2)
            + self.k_top*(self.t - self.t_feeder)*self.vt*self.ds_(3)
            + self.k_lft*(self.t - self.t_amb)*self.vt*self.ds_(4))
        print("k, T", self.k_btm.values(), self.t_feeder.values())
        print("k, T", self.k_rgt.values(), self.t_amb.values())
        print("k, T", self.k_top.values(), self.t_amb.values())
        print("k, T", self.k_lft.values(), self.t_amb.values())

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


if __name__ == "__main__":
    pass
