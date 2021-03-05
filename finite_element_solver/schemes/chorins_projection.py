#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:48 2021

@author: florianma

Simulation gets slower and slower. 1st iteration takes 2s, 10th iteration 30s.
"""
from dolfin import (inner, grad, div, dx, solve, lhs, rhs, ds, dot, nabla_grad,
                    FacetNormal, assemble, outer, Constant, Identity, sym)


def epsilon(u):
    # Define symmetric gradient
    return sym(nabla_grad(u))


def sigma(u, p, mu):
    # Define stress tensor
    return 2*mu*epsilon(u) - p*Identity(len(u))


class ImplicitTentativeVelocityStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        mu = Constant(parameters["viscosity [Pa*s]"])
        dt = Constant(parameters["dt [s]"])
        u, u_1, u_k, vu = domain.u, domain.u_1, domain.u_k, domain.vu
        p_1 = domain.p_1

        n = FacetNormal(domain.mesh)
        acceleration = rho*inner((u-u_1)/dt, vu) * dx
        convection = dot(div(rho*outer(u_k, u)), vu) * dx
        pressure = (inner(p_1, div(vu))*dx - dot(p_1*n, vu)*ds)
        diffusion = (-inner(mu * (grad(u) + grad(u).T), grad(vu))*dx)  # good
        # diffusion = (-inner(mu * (grad(u) + grad(u).T), grad(vu))*dx
        #              + dot(mu * (grad(u) + grad(u).T)*n, vu)*ds)  # very slow!

        F_impl = acceleration + convection + pressure + diffusion
        self.a, self.L = lhs(F_impl), rhs(F_impl)
        self.domain = domain
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        return

    def solve(self):
        bcu = self.domain.bcu
        u_ = self.domain.u_
        u_k = self.domain.u_k

        piccard_iterations = 1
        for k in range(piccard_iterations):
            b = assemble(self.L)
            [bc.apply(b) for bc in bcu]
            solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
            u_k.assign(u_)
        return


class ExplicitTentativeVelocityStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        mu = Constant(parameters["viscosity [Pa*s]"])
        dt = Constant(parameters["dt [s]"])
        u, u_1, p_1, vu = domain.u, domain.u_1, domain.p_1, domain.vu

        n = FacetNormal(domain.mesh)
        acceleration = rho*inner((u-u_1)/dt, vu) * dx
        convection = dot(div(rho*outer(u_1, u)), vu) * dx
        diffusion = (-inner(mu * (grad(u_1) + grad(u_1).T), grad(vu))*dx
                     + dot(mu * (grad(u_1) + grad(u_1).T)*n, vu)*ds)
        pressure = inner(p_1, div(vu))*dx - dot(p_1*n, vu)*ds  # int. by parts
        # TODO: what is better?
        # convection = rho*dot(dot(u_1, nabla_grad(u_k)), vu) * dx
        # diffusion = (mu*inner(grad(u_1), grad(vu))*dx
        #              - mu*dot(nabla_grad(u_1)*n, vu)*ds)  # int. by parts
        F_impl = - acceleration - convection + diffusion + pressure
        self.a, self.L = lhs(F_impl), rhs(F_impl)
        self.domain = domain
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        return

    def solve(self):
        bcu = self.domain.bcu
        u_ = self.domain.u_

        b = assemble(self.L)
        [bc.apply(b) for bc in bcu]
        solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
        return


class PressureStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        dt = Constant(parameters["dt [s]"])
        p, p_1, vp = domain.p, domain.p_1, domain.vp
        p_1, u_ = domain.p_1, domain.u_

        # F = rho/dt * dot(div(u_), vp) * dx + dot(grad(p-p_1), grad(vp)) * dx
        self.a = dot(nabla_grad(p), nabla_grad(vp))*dx
        self.L = (dot(nabla_grad(p_1), nabla_grad(vp))*dx
                  - (rho/dt)*div(u_)*vp*dx)
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcp]
        self.domain = domain
        return

    def solve(self):
        bcp = self.domain.bcp
        p_ = self.domain.p_

        b = assemble(self.L)
        [bc.apply(b) for bc in bcp]
        solve(self.A, p_.vector(), b, 'bicgstab', 'hypre_amg')
        return


class VelocityCorrectionStep():
    def __init__(self, parameters, domain):
        rho = Constant(parameters["density [kg/m3]"])
        dt = Constant(parameters["dt [s]"])
        u, u_, vu = domain.u, domain.u_, domain.vu
        p_1, p_ = domain.p_1, domain.p_

        self.a = dot(u, vu)*dx
        self.L = dot(u_, vu)*dx - dt/rho*dot(nabla_grad(p_ - p_1), vu)*dx
        self.A = assemble(self.a)
        [bc.apply(self.A) for bc in domain.bcu]
        self.domain = domain
        return

    def solve(self):
        bcu = self.domain.bcu
        u_ = self.domain.u_

        b = assemble(self.L)
        [bc.apply(b) for bc in bcu]
        solve(self.A, u_.vector(), b, 'bicgstab', 'hypre_amg')
        return
