#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:48 2021

@author: florianma

"""
from dolfin import (inner, div, dx, solve, lhs, rhs, ds, dot, nabla_grad,
                    FacetNormal, assemble, sym, Identity, Constant)


def epsilon(u):
    # Define symmetric gradient
    return sym(nabla_grad(u))


def sigma(u, p, mu):
    # Define stress tensor
    return 2*mu*epsilon(u) - p*Identity(len(u))


class TentativeVelocityStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        mu = Constant(parameters["viscosity [Pa*s]"])
        dt = Constant(parameters["dt [s]"])
        u, u_1, vu = domain.u, domain.u_1, domain.vu
        p_1 = domain.p_1

        n = FacetNormal(domain.mesh)
        u_mid = (u + u_1) / 2.0
        F1 = rho*dot((u - u_1) / dt, vu)*dx \
            + rho*dot(dot(u_1, nabla_grad(u_1)), vu)*dx \
            + inner(sigma(u_mid, p_1, mu), epsilon(vu))*dx \
            + dot(p_1*n, vu)*ds - dot(mu*nabla_grad(u_mid)*n, vu)*ds
        a1 = lhs(F1)
        L1 = rhs(F1)
        A1 = assemble(a1)
        [bc.apply(A1) for bc in domain.bcu]

        self.a1, self.L1, self.A1 = a1, L1, A1
        self.domain = domain
        return

    def solve(self):
        bcu, u_ = self.domain.bcu, self.domain.u_
        A1, L1 = self.A1, self.L1

        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
        return


class PressureStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        dt = Constant(parameters["dt [s]"])
        p, p_1, vp = domain.p, domain.p_1, domain.vp
        p_1, u_ = domain.p_1, domain.u_

        a2 = dot(nabla_grad(p), nabla_grad(vp))*dx
        L2 = dot(nabla_grad(p_1), nabla_grad(vp))*dx - (rho/dt)*div(u_)*vp*dx
        A2 = assemble(a2)
        [bc.apply(A2) for bc in domain.bcp]

        self.domain = domain
        self.a2, self.A2, self.L2 = a2, A2, L2
        return

    def solve(self):
        bcp, p_ = self.domain.bcp, self.domain.p_
        A2, L2 = self.A2, self.L2

        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
        return


class VelocityCorrectionStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        dt = Constant(parameters["dt [s]"])
        u, u_, vu = domain.u, domain.u_, domain.vu
        p_1, p_ = domain.p_1, domain.p_

        a3 = dot(u, vu)*dx
        L3 = dot(u_, vu)*dx - dt/rho*dot(nabla_grad(p_ - p_1), vu)*dx
        A3 = assemble(a3)

        self.a3, self.A3, self.L3 = a3, A3, L3
        self.domain = domain
        return

    def solve(self):
        u_ = self.domain.u_
        A3, L3 = self.A3, self.L3

        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')
        return
