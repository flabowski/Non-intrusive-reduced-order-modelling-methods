#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:26:14 2021

@author: florianma
"""
import numpy as np
import os
from tqdm import trange  # Progress bar
import matplotlib.pyplot as plt
import dolfin


def create2Dmesh(msh, unused_points):
    """
    Helping function to create a 2D mesh for FEniCS from a gmsh.
    important! Dont leave any unused points like the center of the circle in
    the node list. FEniCS will crash!
    """
    msh.prune_z_0()
    nodes = msh.points[unused_points:]
    cells = msh.cells_dict["triangle"].astype(np.uintp)-unused_points
    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    # point, interval, triangle, quadrilateral, hexahedron
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(nodes))
    editor.init_cells(len(cells))
    [editor.add_vertex(i, n) for i, n in enumerate(nodes)]
    [editor.add_cell(i, n) for i, n in enumerate(cells)]
    editor.close()
    return mesh


def time_stepping(mesh, N, dt, parameter,
                  scheme, solver, plot_res, my_dir, density_varies=False):
    print("snapshots will be saved at: "+my_dir)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
    problem = scheme(mesh, dt, parameter)  # navier_stokes_IPCS
    # store n_snapshots at a time
    n_snapshots = 1000  # per batch
    n_nodes = len(mesh.coordinates())
    print("allocating {:.0f} floats ({:.2f} GB)".format(
        3*n_snapshots*n_nodes, 3*n_snapshots*n_nodes*64/8/1000/1000/1000))
    time = np.zeros((N,))
    u_x = np.zeros((n_snapshots, n_nodes), dtype=np.float64)
    u_y = np.zeros_like(u_x)
    pressure = np.zeros_like(u_x)
    if density_varies:
        density = np.zeros_like(u_x)
    ind = 0

    fig, axs = plot_res(mesh, problem[:5])
    # if density_is_constant:
    #     fig, axs = plot_upr(mesh, u_1, p_1, r_1)
    plt.suptitle("initial condition")
    fn = my_dir+"frame_0.png"
    plt.savefig(fn)
    plt.close(fig)
    for n in trange(N):
        t = n*dt
        res = solver(*problem)  # solve_timestep
        u_1, p_1 = res[0], res[1]
        # save snapshots
        u_x[ind], u_y[ind] = np.split(u_1.compute_vertex_values(mesh), 2, 0)
        pressure[ind] = p_1.compute_vertex_values(mesh)
        if density_varies:
            density[ind] = res[2].compute_vertex_values(mesh)
        time[n] = t
        ind += 1
        if (ind == n_snapshots) or (ind == (N-1)):
            np.save(my_dir+"{:06.0f}u.npy".format(n+1-n_snapshots), u_x)
            np.save(my_dir+"{:06.0f}v.npy".format(n+1-n_snapshots), u_y)
            np.save(my_dir+"{:06.0f}p.npy".format(n+1-n_snapshots), pressure)
            if density_varies:
                np.save(my_dir+"{:06.0f}r.npy".format(n+1-n_snapshots), density)
            ind = 0
        # plot and save snapshots just to see if everything runs smoothly
        if ((n % 100) < 1e-4):
            fig, axs = plot_res(mesh, res)
            # if density_is_constant:
            #     fig, axs = plot_upr(mesh, u_1, p_1, r_1)
            plt.suptitle("t={:.2f} s".format(t))
            fn = my_dir+"frame_{:06.0f}.png".format(n)
            plt.savefig(fn)
            plt.close(fig)

    x, y = np.split(mesh.coordinates(), 2, 1)
    tri = mesh.cells()
    np.save(my_dir+"__t.npy", time)
    np.save(my_dir+"__x.npy", x.ravel())
    np.save(my_dir+"__y.npy", y.ravel())
    np.save(my_dir+"__tri.npy", tri)
    return


def epsilon(u):
    # Define symmetric gradient
    return dolfin.sym(dolfin.nabla_grad(u))


def sigma(u, p, mu):
    # Define stress tensor
    return 2*mu*epsilon(u) - p*dolfin.Identity(len(u))
