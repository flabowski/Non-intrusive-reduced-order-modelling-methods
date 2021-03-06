#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:48 2021

@author: florianma

"""
from dolfin import (dx, solve, lhs, rhs, dot, assemble, grad)


class ConvectionDiffusion():
    def __init__(self, domain):
        D, dt = domain.D, domain.dt
        t, t_1, vt = domain.t, domain.t_1, domain.vt
        u_, boundary_terms = domain.u_, domain.robin_boundary_terms

        # Step 4: Transport of rho / Convection-diffusion and SUPG
        # vr = vr + tau_SUPG * inner(u_, grad(vr))  # SUPG stabilization
        # F4 = dot((t - t_1) / dt, vt)*dx + dot(div(t*u_), vt) * dx       + D*dot(grad(t), grad(vt)) * dx
        # above does not work, below works fine, but is mathematically not correct, since d/dt (rho) is not 0
        F = dot((t - t_1) / dt, vt)*dx + dot(dot(grad(t), u_), vt) * dx \
            + D*dot(grad(t), grad(vt)) * dx + boundary_terms

        a, L = lhs(F), rhs(F)

        A = assemble(a)
        self.a, self.L, self.A = a, L, A
        self.domain = domain
        return

    def solve(self):
        bct, t_ = self.domain.bct, self.domain.t_
        A, L, a = self.A, self.L, self.a
        A = assemble(a)
        b = assemble(L)
        [bc.apply(A) for bc in bct]
        [bc.apply(b) for bc in bct]
        solve(A, t_.vector(), b, 'gmres', 'hypre_amg')
        return
