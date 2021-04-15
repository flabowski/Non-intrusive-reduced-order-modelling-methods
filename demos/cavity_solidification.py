#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 12:29:07 2021

@author: florianma
"""
import numpy as np
from scipy.interpolate import interp1d
from tqdm import trange  # Progress bar
import matplotlib.pyplot as plt
import os
from dolfin import assemble, Expression
import matplotlib as mpl
# mpl.use('Agg')
from finite_element_solver.domains.cavity import (create_cavity_mesh,
                                                  CavityProblemSetup)
from finite_element_solver.schemes.chorins_projection import (
    ImplicitTentativeVelocityStep, PressureStep, VelocityCorrectionStep,
    ExplicitTentativeVelocityStep, TutorialTentativeVelocityStep)
from finite_element_solver.schemes.convection_diffusion import (
    ConvectionDiffusion)


# def body_force():
#     # all the IO and printing happens here
#     my_parameters = {"Diffusivity [-]": 1e-2,
#                      "viscosity solid [Pa*s]": 0.1,  # arbitrary.
#                      "characteristic length [m]": 1.0,
#                      "ambient temperature [°C]": 25,
#                      "initial temperature [°C]": 800,
#                      "temperature feeder [°C]": 800,
#                      "mean velocity lid [m/s]": 1.0,
#                      "gravity [m/s²]": 9.81,
#                      "dt [s]": 0.1
#                      }
#     create_cavity_mesh(lcar=0.02)
#     my_domain = CavityProblemSetup(my_parameters, "mesh.xdmf", "mf.xdmf")

#     cfl = .1
#     dt = cfl*my_domain.mesh.hmin()/my_parameters["mean velocity lid [m/s]"]
#     my_parameters["dt [s]"] = dt
#     my_domain.set_dt(dt)

#     tvs = ImplicitTentativeVelocityStep(my_domain)
#     ps = PressureStep(my_domain)
#     vcs = VelocityCorrectionStep(my_domain)
#     cd = ConvectionDiffusion(my_domain)

#     plot(my_domain.mesh)
#     plt.show()

#     mu_new = mu_Al(my_domain.get_t(), my_parameters["viscosity solid [Pa*s]"])*1000
#     rho_new = rho_Al(my_domain.get_t())/1000
#     my_domain.set_mu(mu_new)
#     my_domain.set_rho(rho_new)

#     my_domain.k_lft.assign(.01)
#     my_domain.k_rgt.assign(.01)
#     my_domain.k_btm.assign(.001)

#     for n in trange(10000):
#         tvs.solve(reassemble_A=True)
#         ps.solve()
#         vcs.solve()
#         cd.solve()

#         my_domain.u_1.assign(my_domain.u_)
#         my_domain.p_1.assign(my_domain.p_)
#         my_domain.t_1.assign(my_domain.t_)
#         mu_new = mu_Al(my_domain.get_t(),
#                        my_parameters["viscosity solid [Pa*s]"])*1000
#         rho_new = rho_Al(my_domain.get_t())/1000
#         my_domain.set_mu(mu_new)
#         my_domain.set_rho(rho_new)
#         if (n % 1) == 0:
#             fig, ax = my_domain.plot()
#             plt.savefig("tst{:.0f}.png".format(n))
#             plt.close()


def Aluminum():
    # all the IO and printing happens here
    # save every nth frame
    T = 1000
    n_save = 2
    n_ts = 10000  # run n_ts timesteps
    factor_rho = 1.
    factor_mu = 1000.
    k = 205  # W/(m K)
    cp = 0.91 * 1000  # kJ/(kg K) *1000 = J/(kg K)
    rho = 2350  # kg /m3
    k_r = 0.001
    alpha = k/(cp*rho)
    my_parameters = {"Diffusivity [-]": alpha,  # Al.: 0.0001
                     "viscosity solid [Pa*s]": 100.,  # arbitrary.
                     "characteristic length [m]": 1.0,
                     "ambient temperature [°C]": 600,
                     "initial temperature [°C]": 670,
                     "temperature feeder [°C]": 670,
                     "thermal conductivity top [W/(m K)]": 0.,
                     "thermal conductivity left [W/(m K)]": 0.,
                     "thermal conductivity bottom [W/(m K)]": 0.,
                     "thermal conductivity right [W/(m K)]": k_r,
                     "mean velocity lid [m/s]": 0.0,  # 0.00001
                     "gravity [m/s²]": 9.81,
                     "dt [s]": .2
                     }
    create_cavity_mesh(lcar=0.02)
    my_domain = CavityProblemSetup(my_parameters, "mesh.xdmf", "mf.xdmf")

    tvs = ImplicitTentativeVelocityStep(my_domain)
    # tvs = ExplicitTentativeVelocityStep(my_domain)
    ps = PressureStep(my_domain, penalization=1e-7)
    vcs = VelocityCorrectionStep(my_domain)
    cd = ConvectionDiffusion(my_domain, k=0.5)

    mu_new = mu_Al(my_domain.get_t(), my_parameters["viscosity solid [Pa*s]"])
    rho_new = rho_Al(my_domain.get_t())
    my_domain.set_mu(mu_new*factor_mu)
    my_domain.set_rho(rho_new*factor_rho)
    my_domain.initial_condition_from_file("u_.xdmf", "p_.xdmf")
    my_domain.u_1.assign(my_domain.u_)
    my_domain.p_1.assign(my_domain.p_)

    rho = np.mean(my_domain.get_rho())
    U = my_parameters["mean velocity lid [m/s]"]
    L = my_parameters["characteristic length [m]"]
    mu = np.mean(my_domain.get_mu())
    Re = rho*U*L/mu
    print("Re = ", Re)
    print("rho = ", rho)
    print("mu = ", mu)
    print("dt = ", my_domain.get_dt())

    my_domain.k_lft.assign(.0)
    my_domain.k_btm.assign(.0)
    my_domain.k_top.assign(.0)
    my_domain.k_rgt.assign(k_r)
    n_nodes = my_domain.p_.compute_vertex_values(my_domain.mesh).size
    velocity = np.zeros((n_ts//n_save, n_nodes*2))
    pressure = np.zeros((n_ts//n_save, n_nodes))
    time = np.zeros((n_ts//n_save,))
    temperature = np.zeros((n_ts//n_save, n_nodes))
    tri = my_domain.mesh.cells()
    x, y = np.split(my_domain.mesh.coordinates(), 2, 1)
    u, v = np.split(velocity, 2, 0)
    x, y, u, v = x.ravel(), y.ravel(), u.ravel(), v.ravel()

    ds_r = my_domain.ds_(my_domain.bc_dict["right"])
    A = assemble(Expression("1", degree=1) * ds_r)
    # [600,   400,   425,  450,   475,  500,   525,  550,   575,  600,   625,  650]
    # [0.07, 0.27, 0.245, 0.22, 0.195, 0.17, 0.145, 0.12, 0.095, 0.07, 0.045, 0.02]
    for dt in [10, 25, 50, 75, 100, 150, 200, 250, 300]:
        my_dir = "../doc/cavity_solidification_dt({:.0f})/".format(dt)
        print("snapshots will be saved at: "+my_dir)
        if not os.path.exists(my_dir):
            os.makedirs(my_dir)
        fig, ax = my_domain.plot()
        plt.savefig(my_dir+"tst0_init.png", dpi=150)
        plt.close()
        for n in trange(n_ts):
            # get mean temperature on the right wall (inside):
            # integrate temperature at the right wall and divide by the area
            t_r = assemble(my_domain.t_*ds_r) / A
            t_amb = t_r -dt
            my_domain.t_amb.assign(t_amb)

            tvs.solve(reassemble_A=True)
            ps.solve()
            vcs.solve()
            cd.solve()

            my_domain.u_1.assign(my_domain.u_)
            my_domain.p_1.assign(my_domain.p_)
            my_domain.t_1.assign(my_domain.t_)
            mu_new = mu_Al(my_domain.get_t(),
                           my_parameters["viscosity solid [Pa*s]"])
            rho_new = rho_Al(my_domain.get_t())
            my_domain.set_mu(mu_new*factor_mu)
            my_domain.set_rho(rho_new*factor_rho)
            # print(np.min(my_domain.get_t()), np.max(my_domain.get_t()))
            # print(np.min(my_domain.get_rho()), np.max(my_domain.get_rho()))
            if ((n % 100) == 0) or (n < 10):
                fig, ax = my_domain.plot()
                plt.savefig(my_dir+"tst{:.0f}.png".format(n), dpi=200)
                plt.close()

            if (n % n_save) == 0:
                i = n // n_save
                mesh = my_domain.mesh
                velocity[i] = my_domain.u_.compute_vertex_values(mesh)
                pressure[i] = my_domain.p_.compute_vertex_values(mesh)
                temperature[i] = my_domain.t_.compute_vertex_values(mesh)
                time[i] = n*my_domain.get_dt()
            # if n == 2000:
            #     f_out = df.XDMFFile("u_.xdmf")
            #     # timestep 0, Not appending to file
            #     f_out.write_checkpoint(my_domain.u_, "f", 0,
            #                            df.XDMFFile.Encoding.HDF5, False)
            #     f_out.close()
            #     f_out = df.XDMFFile("p_.xdmf")
            #     f_out.write_checkpoint(my_domain.p_, "f", 0,
            #                            df.XDMFFile.Encoding.HDF5, False)
            #     f_out.close()Tamb({:.0f})/".format(t_amb)
        pf = "Tamb{:.0f}_".format(t_amb)  # prefix
        np.save(my_dir+pf+"time.npy", time)
        np.save(my_dir+pf+"x.npy", x.ravel())
        np.save(my_dir+pf+"y.npy", y.ravel())
        np.save(my_dir+pf+"tri.npy", tri)
        np.save(my_dir+pf+"velocity.npy", velocity)
        np.save(my_dir+pf+"pressure.npy", pressure)
        np.save(my_dir+pf+"temperature.npy", temperature)

        t_init = my_parameters["initial temperature [°C]"]
        my_domain.t_.vector().vec().array[:] = t_init
        my_domain.t_1.vector().vec().array[:] = t_init
        mu_new = mu_Al(my_domain.get_t(),
                       my_parameters["viscosity solid [Pa*s]"])
        rho_new = rho_Al(my_domain.get_t())
        my_domain.set_mu(mu_new*factor_mu)
        my_domain.set_rho(rho_new*factor_rho)
        my_domain.initial_condition_from_file("u_.xdmf", "p_.xdmf")
        my_domain.u_1.assign(my_domain.u_)
        my_domain.p_1.assign(my_domain.p_)


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
    f_rho = interp1d(temperature, rho_al100, kind='linear', bounds_error=False,
                     fill_value="extrapolate")
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
    mu_arr = np.zeros(T.shape)
    x = np.array([25, 570, 620, 640, 650, 655.00001])
    y = np.array([0.0, 0., 0.1, 0.2, 0.5, 1.0])
    f_xl = interp1d(x, y, kind='linear', bounds_error=False,
                    fill_value="extrapolate")
    # if np.sum(PARTIALLY_SOLID) > 0:
    #     print("f_xl: ", np.min(T[PARTIALLY_SOLID]), np.max(T[PARTIALLY_SOLID]))
    xl = f_xl(T[PARTIALLY_SOLID])

    x = np.array([-1e-6, 1.00000001])
    y = np.array([mu_solid, mu_liquidus])
    f_mux = interp1d(x, y, kind='linear', bounds_error=False,
                     fill_value="extrapolate")
    mu_arr[PARTIALLY_SOLID] = f_mux(xl)

    temp = np.array([654.99999, 700, 800, 900, 1000, 1100])
    nu = np.array([4.99, 4.11, 3.7, 3.36, 3.1]) * 1e-7
    mu_table = np.r_[nu_liquidus, nu]*rho_Al(temp)
    f_mu = interp1d(temp, mu_table, kind='linear', bounds_error=False,
                    fill_value="extrapolate")
    mu_arr[~PARTIALLY_SOLID] = f_mu(T[~PARTIALLY_SOLID])
    # mu_arr[:] = mu_solid
    return mu_arr


if __name__ == "__main__":
    Aluminum()
