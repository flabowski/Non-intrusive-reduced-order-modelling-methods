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
from dolfin import (VectorElement, FiniteElement, inner, grad, dx, div, solve,
                    lhs, rhs, split, project, dot)
# from mpi4py import MPI
# import meshio
# from finite_element_solver.domains.cylinder import create_entity_mesh
# from meshing.cavity import mesh
# TODO: make a generic module for create_entity_mesh, plot_mesh




class CavityProblemSetup():
    def __init__(self, parameters, mesh_name, facet_name):
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
        mf = self.mf = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        self.bc_dict = {"bottom": 1, "right": 2, "top": 3, "left": 4}

        T_init = Constant(parameters["initial temperature [°C]"])
        self.t_amb = Constant(parameters["ambient temperature [°C]"])
        self.t_feeder = Constant(parameters["temperature feeder [°C]"])
        self.k_top = Constant(parameters["thermal conductivity top [W/(m K)]"])
        self.k_lft = Constant(parameters["thermal conductivity left [W/(m K)]"])
        self.k_btm = Constant(parameters["thermal conductivity bottom [W/(m K)]"])
        self.k_rgt = Constant(parameters["thermal conductivity right [W/(m K)]"])
        self.U_m = Constant(parameters["mean velocity lid [m/s]"])
        g = parameters["gravity [m/s²]"]
        self.g = Constant((0.0, -g))
        self.dt = Constant(parameters["dt [s]"])
        self.D = Constant(parameters["Diffusivity [-]"])

        self.V = V = VectorFunctionSpace(self.mesh, 'P', 2)
        self.Q = Q = FunctionSpace(self.mesh, 'P', 1)
        self.T = T = FunctionSpace(self.mesh, 'P', 1)

        self.ds_ = Measure("ds", domain=self.mesh, subdomain_data=mf)
        self.vu, self.vp, self.vt = (TestFunction(V), TestFunction(Q),
                                     TestFunction(T))
        self.u_, self.p_, self.t_ = Function(V), Function(Q), Function(T)
        self.mu, self.rho = Function(T), Function(T)
        self.u_1, self.p_1, self.t_1, self.rho_1 = (Function(V), Function(Q),
                                                    Function(T), Function(T))
        self.u, self.p, self.t = (TrialFunction(V), TrialFunction(Q),
                                  TrialFunction(T))

        # boundary conditions
        self.no_slip = Constant((0., 0))
        self.topflow = Expression(("-x[0] * (x[0] - 1.0) * 6.0 * m", "0.0"),
                                  m=self.U_m, degree=2)
        bc0 = DirichletBC(V, self.topflow, mf, self.bc_dict["top"])
        bc1 = DirichletBC(V, self.no_slip, mf, self.bc_dict["left"])
        bc2 = DirichletBC(V, self.no_slip, mf, self.bc_dict["bottom"])
        bc3 = DirichletBC(V, self.no_slip, mf, self.bc_dict["right"])
        # bc4 = df.DirichletBC(Q, df.Constant(0), top)
        # bc3 = df.DirichletBC(T, df.Constant(800), top)
        self.bcu = [bc0, bc1, bc2, bc3]
        self.bcp = [DirichletBC(Q, Constant(0), mf, self.bc_dict["top"])]
        self.bcp = []
        self.bct = []
        # self.bct = [DirichletBC(T, Constant(self.t_feeder), mf,
        #                         self.bc_dict["top"])]

        self.robin_boundary_terms = (
            self.k_btm*(self.t - self.t_amb)*self.vt*self.ds_(1)
            + self.k_rgt*(self.t - self.t_amb)*self.vt*self.ds_(2)
            # + self.k_top*(self.t - self.t_feeder)*self.vt*self.ds_(3)
            + self.k_lft*(self.t - self.t_amb)*self.vt*self.ds_(4))
        print("k, T", self.k_btm.values(), self.t_feeder.values())
        print("k, T", self.k_rgt.values(), self.t_amb.values())
        # print("k, T", self.k_top.values(), self.t_amb.values())
        print("k, T", self.k_lft.values(), self.t_amb.values())

        # set initial values
        # TODO: find a better solution
        x, y = T.tabulate_dof_coordinates().T
        self.u_1.vector().vec().array[:] = 1e-6
        # self.u_k.vector().vec().array[:] = 1e-6
        self.p_.vector().vec().array[:] = -self.rho.vector().vec().array*g*y
        self.p_1.vector().vec().array[:] = -self.rho.vector().vec().array*g*y
        self.t_1.vector().vec().array = T_init-x*10
        self.t_.assign(self.t_1)
        return

    def stokes(self):
        P2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        TH = P2 * P1
        VQ = FunctionSpace(self.mesh, TH)
        mf = self.mf
        self.no_slip = Constant((0., 0))
        self.topflow = Expression(("-x[0] * (x[0] - 1.0) * 6.0 * m", "0.0"),
                                  m=self.U_m, degree=2)
        bc0 = DirichletBC(VQ.sub(0), self.topflow, mf, self.bc_dict["top"])
        bc1 = DirichletBC(VQ.sub(0), self.no_slip, mf, self.bc_dict["left"])
        bc2 = DirichletBC(VQ.sub(0), self.no_slip, mf, self.bc_dict["bottom"])
        bc3 = DirichletBC(VQ.sub(0), self.no_slip, mf, self.bc_dict["right"])
        # bc4 = DirichletBC(VQ.sub(1), Constant(0), mf, self.bc_dict["top"])
        bcs = [bc0, bc1, bc2, bc3]

        vup = TestFunction(VQ)
        up = TrialFunction(VQ)
        # the solution will be in here:
        up_ = Function(VQ)

        u, p = split(up)  # Trial
        vu, vp = split(vup)  # Test
        u_, p_ = split(up_)  # Function holding the solution
        F = self.mu*inner(grad(vu), grad(u))*dx - inner(div(vu), p)*dx \
            - inner(vp, div(u))*dx - dot(self.g*self.rho, vu)*dx
        solve(lhs(F) == rhs(F), up_, bcs=bcs)
        self.u_.assign(project(u_, self.V))
        self.p_.assign(project(p_, self.Q))
        return

    def initial_condition_from_file(self, path_u, path_p):
        # path_u, path_p = "../u_.xdmf", "../p_.xdmf"
        f_in = XDMFFile(path_u)
        f_in.read_checkpoint(self.u_, "f", 0)
        f_in = XDMFFile(path_p)
        f_in.read_checkpoint(self.p_, "f", 0)
        return

    def get_rho(self):
        return self.rho.vector().vec().array

    def set_rho(self, rho):
        self.rho.vector().vec().array[:] = rho

    def get_mu(self):
        return self.mu.vector().vec().array

    def set_mu(self, mu):
        self.mu.vector().vec().array[:] = mu

    def get_t(self):
        return self.t_.vector().vec().array

    def set_t(self, t):
        self.t_.vector().vec().array[:] = t

    def get_dt(self):
        return self.dt.values()

    def set_dt(self, dt):
        self.dt.assign(dt)

    def get_D(self):
        return self.D.values()

    def set_D(self, D):
        self.D.assign(D)

    def get_t_amb(self):
        return self.t_amb.values()

    def set_t_amb(self, t_amb):
        self.t_amb.assign(t_amb)

    def plot(self):
        cmap = mpl.cm.inferno
        cmap_r = mpl.cm.inferno_r
        u, p, t, m, r = self.u_, self.p_, self.t_, self.mu, self.rho
        mesh = self.mesh
        w0 = u.compute_vertex_values(mesh)
        w0.shape = (2, -1)
        magnitude = np.linalg.norm(w0, axis=0)*100
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
        ax1.plot(x, y, "k.", ms=.5)
        not0 = magnitude > 1e-6
        if np.sum(not0) > 0:
            ax1.quiver(x[not0], y[not0], u[not0], v[not0], magnitude[not0])
        c2 = ax2.tricontourf(x, y, tri, pressure, levels=40, cmap=cmap)
        c3 = ax3.tricontourf(x, y, tri, temperature, levels=40, cmap=cmap,
                             vmin=610.0, vmax=660.0
                             )
        # print(viscosity)
        c4 = ax4.tricontourf(x, y, tri, viscosity, levels=40,
                              # vmin=1, vmax=3,
                             cmap=cmap_r)
        c5 = ax5.tricontourf(x, y, tri, density, levels=40,
                              # vmin=self.rho(800.), vmax=self.rho(600.),
                             cmap=cmap_r)
        plt.colorbar(c2, ax=ax2)
        plt.colorbar(c3, ax=ax3)
        plt.colorbar(c4, ax=ax4)
        plt.colorbar(c5, ax=ax5)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax3.set_aspect("equal")
        ax4.set_aspect("equal")
        ax5.set_aspect("equal")
        ax1.set_title("velocity\n{:.4f} ... {:.5f} cm/s".format(
            magnitude.min(), magnitude.max()))
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
