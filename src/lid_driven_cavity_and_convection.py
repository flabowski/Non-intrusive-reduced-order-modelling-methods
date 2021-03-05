#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:51:59 2021

@author: florianma
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pygmsh
import timeit
from tqdm import trange  # Progress bar
import dolfin as df
import matplotlib as mpl
from common import time_stepping, create2Dmesh, sigma, epsilon
from scipy.interpolate import interp1d


def navier_stokes_IPCS_cavity(mesh, dt, parameter):
    """
    fenics code: weak form of the problem.
    """
    dx, ds = df.dx, df.ds
    dot, inner, outer, div = df.dot,  df.inner,  df.outer, df.div
    nabla_grad, grad = df.nabla_grad, df.grad
    test_f, trial_f = df.TestFunction, df.TrialFunction

    U0, D, mu_solid = parameter
    g = 9.81/1.0000000000
    # function space
    V = df.VectorFunctionSpace(mesh, 'P', 2)
    Q = df.FunctionSpace(mesh, 'P', 1)
    T = df.FunctionSpace(mesh, 'P', 1)

    ASD1 = df.AutoSubDomain(top)
    ASD2 = df.AutoSubDomain(left)
    ASD3 = df.AutoSubDomain(bottom)
    ASD4 = df.AutoSubDomain(right)
    mf = df.MeshFunction("size_t", mesh, 1)
    mf.set_all(9999)
    ASD1.mark(mf, 1)
    ASD2.mark(mf, 2)
    ASD3.mark(mf, 3)
    ASD4.mark(mf, 4)
    ds_ = ds(subdomain_data=mf, domain=mesh)
    print(np.unique(ds_(3).subdomain_data().array()))

    vu, vp, vt = test_f(V), test_f(Q), test_f(T)
    u_, p_, t_ = df.Function(V), df.Function(Q), df.Function(T)  # solution
    mu_k, rho_k = df.Function(T), df.Function(T)
    u_1, p_1, t_1, rho_1 = df.Function(V), df.Function(Q), df.Function(T), df.Function(T)  # solution1
    u, p, t = trial_f(V), trial_f(Q), trial_f(T)  # unknown!
    u_k = df.Function(V)

    # boundary conditions
    no_slip = df.Constant((0., 0))
    topflow = df.Expression(("-x[0] * (x[0] - 1.0) * 6.0 * m", "0.0"),
                            m=U0, degree=2)
    bc0 = df.DirichletBC(V, topflow, top)
    bc1 = df.DirichletBC(V, no_slip, left)
    bc2 = df.DirichletBC(V, no_slip, bottom)
    bc3 = df.DirichletBC(V, no_slip, right)
    # bc4 = df.DirichletBC(Q, df.Constant(0), top)
    # bc3 = df.DirichletBC(T, df.Constant(800), top)
    bcu = [bc0, bc1, bc2, bc3]
    # no boundary conditions for the pressure
    bcp = []
    # bcp = [df.DirichletBC(Q, df.Constant(0), top)]
    bct = []

    # set initial temp: 500°C  y=0, 800°C at y=1
    x, y = np.split(T.tabulate_dof_coordinates(), 2, 1)
    u_1.vector().vec().array[:] = 1e-6
    u_k.vector().vec().array[:] = 1e-6
    p_.vector().vec().array[:] = -rho(750)*g*y.ravel()
    p_1.vector().vec().array[:] = -rho(750)*g*y.ravel()
    t_1.vector().vec().array = (y.ravel())*100 + 700
    t_.assign(t_1)
    mu_k.vector().vec().array = mu(t_1.vector().vec().array, mu_solid)
    rho_k.vector().vec().array = rho(t_1.vector().vec().array)
    rho_1.vector().vec().array = rho(t_1.vector().vec().array)

    n = df.FacetNormal(mesh)

    # implicit:
    acceleration = inner((rho_k*u - rho_1*u_1)/dt, vu) * dx
    convection = dot(div(rho_k*outer(u_k, u)), vu) * dx
    pressure = inner(p_1, div(vu))*dx - dot(p_1*n, vu)*ds  # integrated by parts
    diffusion = -inner(mu_k * (grad(u) + grad(u).T), grad(vu))*dx \
                + dot(mu_k * (grad(u) + grad(u).T)*n, vu)*ds  # integrated by parts
    body_force = dot(df.Constant((0.0, -g))*rho_k, vu)*dx \
               + dot(df.Constant((0.0, 0.0)), vu) * ds
    F1 = -acceleration - convection + diffusion + pressure + body_force
    a1, L1 = df.lhs(F1), df.rhs(F1)

    # Define variational problem for step 2
    F2 = rho_k / dt * dot(div(u_), vp) * dx + dot(grad(p-p_1), grad(vp)) * dx  # grad(p-p_1)/2 * grad(vp) * dx does not work
    a2, L2 = df.lhs(F2), df.rhs(F2)

    # Define variational problem for step 3, where u_ = u* from step 1
    F3 = -rho_k / dt * dot(u-u_, vu) * dx - dot(grad(p_-p_1), vu) * dx
    a3, L3 = df.lhs(F3), df.rhs(F3)

    # Step 4: Transport of rho / Convection-diffusion and SUPG
    # vr = vr + tau_SUPG * inner(u_, grad(vr))  # SUPG stabilization
    # F4 = dot((t - t_1) / dt, vt)*dx + dot(div(t*u_), vt) * dx       + D*dot(grad(t), grad(vt)) * dx
    # above does not work, below works fine, but is mathematically not correct, since d/dt (rho) is not 0
    F4 = dot((t - t_1) / dt, vt)*dx + dot(dot(grad(t), u_), vt)*dx \
        + D*dot(grad(t), grad(vt)) * dx
    a4, L4 = df.lhs(F4), df.rhs(F4)

    # Robin BC: HT on the walls. ht coefficient k is arbitray
    t_amb, t_feeder = 100., 800.
    k_top, k_lft, k_btm, k_rgt = (1e-3, 3.33e-4, 3.33e-4, 3.33e-4)
    F4 += k_top*(t - t_feeder)*vt*ds_(1)
    F4 += k_lft*(t - t_amb)*vt*ds_(2)
    F4 += k_btm*(t - t_amb)*vt*ds_(3)
    F4 += k_rgt*(t - t_amb)*vt*ds_(4)

    # Assemble matrices
    A1 = df.assemble(a1)
    A2 = df.assemble(a2)
    A3 = df.assemble(a3)
    # A4 = assemble(a4)
    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]
    [bc.apply(A3) for bc in bcu]
    return (u_1, p_1, t_1, mu_k, rho_k, u_, p_, t_, u_k, D,
            L1, a1, L2, A2, L3, A3, L4, a4, bcu, bcp, bct)


def solve_timestep(u_1, p_1, t_1, mu_k, rho_k, u_, p_, t_, u_k, D,
                   L1, a1, L2, A2, L3, A3, L4, a4, bcu, bcp, bct):
    assemble = df.assemble
    solve = df.solve
    # Step 1: Tentative velocity step
    for k in range(4):
        A1 = assemble(a1)   # needs to be reassembled because viscosity changed!
        b1 = assemble(L1)
        [bc.apply(A1) for bc in bcu]
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
        # res = np.sum((u_k.compute_vertex_values(mesh) - u_.compute_vertex_values(mesh))**2)
        # print(k, res)
        # plot_upt(mesh, (u_, p_, t_1, mu_k, rho_k))
        # plt.suptitle(1)
        # plt.show()
        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
        # plot_upt(mesh, (u_, p_, t_1, mu_k, rho_k))
        # plt.suptitle(2)
        # plt.show()
        # Step 3: Velocity correction step
        b3 = assemble(L3)
        [bc.apply(b3) for bc in bcu]
        solve(A3, u_.vector(), b3, 'cg', 'sor')
        # plot_upt(mesh, (u_, p_, t_1, mu_k, rho_k))
        # plt.suptitle(3)
        # plt.show()
        # Step 4: Transport of T
        A4 = assemble(a4)
        b4 = assemble(L4)
        [bc.apply(A4) for bc in bct]
        [bc.apply(b4) for bc in bct]
        solve(A4, t_.vector(), b4, 'gmres', 'hypre_amg')

        u_k.assign(u_)
        mu_k.vector().vec().array = mu(t_.vector().vec().array, mu_solid)
        rho_k.vector().vec().array = rho(t_.vector().vec().array)

        # plot_upt(mesh, (u_k, p_, t_, mu_k, rho_k))
    #     plt.suptitle(4)
    #     plt.show()
    # asd
    # Update previous solution
    u_1.assign(u_)
    p_1.assign(p_)
    t_1.assign(t_)
    # plot_upt(mesh, (u_, p_, t_, mu_k, rho_k))
    # plt.suptitle(4)
    # plt.show()
    return u_1, p_1, t_1, mu_k, rho_k


def cavity(lcar, L):
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
    mesh = create2Dmesh(msh, 0)
    return mesh


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


def plot_upt(mesh, res):
    cmap = mpl.cm.inferno
    cmap_r = mpl.cm.inferno_r
    u, p, t, m, r = res
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
    # fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
    # [[ax1, ax2, ax3], [ax4, ax5, ax6]] = axs
    ax1.quiver(x, y, u, v, magnitude)
    c2 = ax2.tricontourf(x, y, tri, pressure, levels=40, cmap=cmap)
    c3 = ax3.tricontourf(x, y, tri, temperature, levels=40,
                         vmin=600., vmax=800., cmap=cmap)
    c4 = ax4.tricontourf(x, y, tri, viscosity, levels=40,
                         vmin=mu(800, .1), vmax=mu(600, .1), cmap=cmap_r)
    c5 = ax5.tricontourf(x, y, tri, density, levels=40,
                         vmin=rho(800.), vmax=rho(600.), cmap=cmap_r)
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


def rho(T):
    # print("rho: ", np.min(T), np.max(T))
    return rho_Al(T)


def mu(T, mu_solid):
    # print("mu: ", np.min(T), np.max(T))
    return mu_Al(T, mu_solid)


def rho_Al(T):
    """
    see: https://www.epj-conferences.org/articles/epjconf/pdf/2011/05/epjconf_lam14_01024.pdf
    Table 4 in Viscosity and volume properties of the Al-Cu melts.
    N. Konstantinova, A. Kurochkin, and P. Popel
    """
    temperature = np.array([0.00, 700., 750., 800., 850., 900.,
                            950., 1000, 1050, 1100, 1150, 1200,
                            1250, 1300, 1350, 1400, 1450, 1500])
    rho_al100 = np.array([2380.0, 2351.5, 2340.6, 2329.8, 2318.9, 2308.1,
                          2297.2, 2286.3, 2275.5, 2264.6, 2253.8, 2242.9,
                          2232.1, 2221.2, 2210.4, 2199.5, 2188.6, 2177.8])
    # [2875.6, 2863.4, 2851.2, 2839.1, 2826.9, 2814.7, 2802.5, 2790.3, 2778.2,
    # 2766.0, 2753.8, 2741.6, 2729.4, 2717.3, 2705.1, 2692.9, 2680.7]
    # [3266.9, 3248.4, 3230.0, 3211.6, 3193.1, 3174.7, 3156.2, 3137.8, 3119.3,
    # 3100.9, 3082.5, 3064.0, 3045.6, 3027.1, 3008.7, 2990.2, 2971.8]
    # [3353.2, 3333.3, 3313.4, 3293.5, 3273.6, 3253.6, 3233.7, 3213.8, 3193.9,
    # 3174.0, 3154.1, 3134.2, 3114.3, 3094.4, 3074.5, 3054.6, 3034.7]
    f_rho = interp1d(temperature, rho_al100, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f_rho(T)  # kg/m3


def mu_Al(T, mu_solid):
    """
    see: https://www.epj-conferences.org/articles/epjconf/pdf/2011/05/epjconf_lam14_01024.pdf
    Table 1 in Viscosity and volume properties of the Al-Cu melts.
    N. Konstantinova, A. Kurochkin, and P. Popel
    Honey at 20°: 10000 mPa
    Water: 0.89 mPa
    Milk: 2.12 mPa
    Aluminum (100 % liquid at 655°): 1.3 mPa
    """
    if not isinstance(T, np.ndarray):
        T = np.array([T], dtype=np.float64)
    # mu_solid = 2.12 / 1000.  # defined in main

    T_liquidus = 655.
    rho_liquidus = rho_Al(T_liquidus)  # kg/m3
    mu_liquidus = 1.3 / 1000.  # = 0.00123 Pa s = 1.3 mPa s
    nu_liquidus = mu_liquidus/rho_liquidus  # 5.52 m2/s

    PARTIALLY_SOLID = T < T_liquidus
    mu_arr = np.zeros_like(T)
    x = np.array([25, 570, 620, 640, 650, 655.00001])
    y = np.array([0.0, 0., 0.1, 0.2, 0.5, 1.0])
    f_xl = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
    # if np.sum(PARTIALLY_SOLID) > 0:
    #     print("f_xl: ", np.min(T[PARTIALLY_SOLID]), np.max(T[PARTIALLY_SOLID]))
    xl = f_xl(T[PARTIALLY_SOLID])

    x = np.array([-1e-6, 1.00000001])
    y = np.array([mu_solid, mu_liquidus])
    f_mux = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
    mu_arr[PARTIALLY_SOLID] = f_mux(xl)

    temp = np.array([654.99999, 700, 800, 900, 1000, 1100])
    nu = np.array([4.99, 4.11, 3.7, 3.36, 3.1]) * 1e-7
    mu_table = np.r_[nu_liquidus, nu]*rho_Al(temp)
    f_mu = interp1d(temp, mu_table, kind='linear', bounds_error=False, fill_value="extrapolate")
    mu_arr[~PARTIALLY_SOLID] = f_mu(T[~PARTIALLY_SOLID])
    # mu_arr[:] = mu_solid
    return mu_arr


if __name__ == "__main__":
    # cfl = .05
    # T_end = 4000
    # N = int((T_end/dt) // 1)
    dt = .01
    N = 6000

    L = 1.0
    mesh = cavity(.02, L)
    df.plot(mesh)
    Re = .4
    D = 1e-2  # diffusion coeff.

    for mu_solid in np.array([100., 200, 500, 1000, 1500])/1000:
        U0 = (Re/(rho(700)*L)*mu(700, mu_solid))[0]
        nu = mu(700, mu_solid)/rho(700)
        # dt = cfl*mesh.hmin()/U0
        print("rho:", rho(700))
        print("mu:", mu(700, mu_solid)[0])
        print("U0:", U0)
        print("dt:", dt)
        parameter = [U0, D, mu_solid]
        my_dir = "../doc/cavity/mu_{}/".format(mu_solid*1000)
        print(Re, my_dir)

        print(dt)
        print("Re set to: ", rho(700)*U0*L/mu(700, mu_solid))
        print("Re set to: ", U0*L/nu)
        print("cfl number: ", np.mean(U0)*dt/mesh.hmin())
        print(N, "timesteps")
        print("Unknowns: ", mesh.num_edges())
        print("coordinates: ", len(mesh.coordinates()))

        tic = timeit.default_timer()
        time_stepping(mesh, N, dt, parameter, navier_stokes_IPCS_cavity,
                      solve_timestep, plot_upt, my_dir, density_varies=True)
        toc = timeit.default_timer()

        print("time IPCS:", toc-tic)
        print("-----------------------end-of-iteration-----------------------")
