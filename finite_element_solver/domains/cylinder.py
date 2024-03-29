#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:51:13 2021

@author: florianma
"""

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
# import pygmsh
# import meshio
# from mpi4py import MPI
# from meshing.cylinder import mesh


class SimpleMesh():
    """
    Simple representation of mesh as points (geometry) and connecitivities (topology)
    Used for
    """

    def __init__(self, dolfin_mesh):
        self.points = dolfin_mesh.coordinates()
        self.simplices = dolfin_mesh.cells()


def plot(mesh):
    """
    2D plot of mesh
    """
    from scipy.spatial import delaunay_plot_2d
    fig = delaunay_plot_2d(SimpleMesh(mesh))
    ax = fig.gca()
    ax.set_aspect("equal")
    return fig, ax


class ChannelProblemSetup():
    def __init__(self, parameters, mesh_name, facet_name,
                 bc_dict={"obstacle": 2, "channel_walls": 1, "inlet": 3,
                          "outlet": 4}):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        self.mesh = df.Mesh()
        with df.XDMFFile(mesh_name) as infile:
            infile.read(self.mesh)

        mvc = df.MeshValueCollection("size_t", self.mesh,
                                     self.mesh.topology().dim() - 1)
        with df.XDMFFile(facet_name) as infile:
            infile.read(mvc, "name_to_read")
        mf = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        V = df.VectorFunctionSpace(self.mesh, 'P', 2)
        Q = df.FunctionSpace(self.mesh, 'P', 1)
        self.rho = df.Constant(parameters["density [kg/m3]"])
        self.mu = df.Constant(parameters["viscosity [Pa*s]"])
        self.dt = df.Constant(parameters["dt [s]"])
        self.g = df.Constant((0, 0))
        self.vu, self.vp = df.TestFunction(V), df.TestFunction(Q)
        self.u_, self.p_ = df.Function(V), df.Function(Q)
        self.u_1, self.p_1 = df.Function(V), df.Function(Q)
        self.u_k, self.p_k = df.Function(V), df.Function(Q)
        self.u, self.p = df.TrialFunction(V), df.TrialFunction(Q)  # unknown!

        U_m = parameters["velocity [m/s]"]
        x = [0, .41 / 2]  # center of the channel
        Ucenter = 4.*U_m*x[1]*(.41-x[1])/(.41*.41)
        U0_str = "4.*U_m*x[1]*(.41-x[1])/(.41*.41)"
        self.U_mean = np.mean(2 / 3 * Ucenter)

        U0 = df.Expression((U0_str, "0"), U_m=U_m, degree=2)
        bc0 = df.DirichletBC(V, df.Constant((0, 0)), mf, bc_dict["obstacle"])
        bc1 = df.DirichletBC(V, df.Constant((0, 0)), mf, bc_dict["channel_walls"])
        bc2 = df.DirichletBC(V, U0, mf, bc_dict["inlet"])
        bc3 = df.DirichletBC(Q, df.Constant(0), mf, bc_dict["outlet"])
        self.bcu = [bc0, bc1, bc2]
        self.bcp = [bc3]
        self.ds_ = df.Measure("ds", domain=self.mesh, subdomain_data=mf)
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

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,
                                       figsize=(12, 6))
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, pressure, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)


if __name__ == "__main__":
    my_parameters = {"density [kg/m3]": 1.0,
                     "viscosity [Pa*s]": 1e-3,
                     "characteristic length [m]": .1,
                     "velocity [m/s]": 1.5,
                     "dt [s]": 0.1
                     }
    create_channel_mesh(lcar=0.02)
    my_domain = ChannelProblemSetup(my_parameters, "mesh.xdmf", "mf.xdmf")
    plot(my_domain.mesh)
    bc_dict = {"obstacle": 2,
               "channel_walls": 1,
               "inlet": 3,
               "outlet": 4}
    print(bc_dict)
    ds_r = my_domain.ds_(bc_dict["channel_walls"])
    Area = df.assemble(df.Expression("1", degree=1) * ds_r)

    # create_channel_mesh(lcar=0.02)
    # my_domain = ChannelProblemSetup(my_parameters, "mesh.xdmf", "mf.xdmf")
