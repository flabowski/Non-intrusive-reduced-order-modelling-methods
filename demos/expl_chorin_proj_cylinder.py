#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:47:33 2021

@author: florianma
"""
from tqdm import trange  # Progress bar
import matplotlib.pyplot as plt
from finite_element_solver.domains.cylinder import create_channel_mesh, ChannelProblemSetup, plot
from finite_element_solver.schemes.chorins_projection import (
    ImplicitTentativeVelocityStep, ExplicitTentativeVelocityStep, PressureStep,
    VelocityCorrectionStep)
from finite_element_solver.schemes.chorins_projection_tutorial import (
    TentativeVelocityStep, PressureStep, VelocityCorrectionStep)
from dolfin import XDMFFile


def test():
    # all the IO and printing happens here
    my_parameters = {"density [kg/m3]": 1.0,
                     "viscosity [Pa*s]": 1e-3,
                     "characteristic length [m]": .1,
                     "velocity [m/s]": 1.5,
                     "dt [s]": 0.1
                     }
    # Create mesh
    create_channel_mesh(lcar=0.02)

    my_domain = ChannelProblemSetup(my_parameters["velocity [m/s]"], "mesh.xdmf", "mf.xdmf")

    cfl = .05
    dt = cfl * my_domain.mesh.hmin() / my_domain.U_mean
    my_parameters["dt [s]"] = dt

    ps = PressureStep(my_parameters, my_domain)
    vcs = VelocityCorrectionStep(my_parameters, my_domain)
    tvs = TentativeVelocityStep(my_parameters, my_domain)
    tvs = ImplicitTentativeVelocityStep(my_parameters, my_domain)
    tvs = ExplicitTentativeVelocityStep(my_parameters, my_domain)

    plot(my_domain.mesh)
    plt.show()

    rho = my_parameters["density [kg/m3]"]
    U = my_domain.U_mean
    L = my_parameters["characteristic length [m]"]
    mu = my_parameters["viscosity [Pa*s]"]
    Re = rho * U * L / mu
    print("Re = ", Re)
    print("rho = ", rho)
    print("mu = ", mu)
    print("dt = ", dt)
    outfile = XDMFFile("u.xdmf")
    append = False
    for n in trange(8000):
        tvs.solve()
        ps.solve()
        vcs.solve()

        my_domain.u_1.assign(my_domain.u_)
        my_domain.p_1.assign(my_domain.p_)
        outfile.write_checkpoint(my_domain.u_, "u", n * dt, append=append)
        append = True
        if (n % 100) == 0:
            fig, ax = my_domain.plot()
            plt.savefig("tst.png")
            plt.close()


if __name__ == "__main__":
    test()
