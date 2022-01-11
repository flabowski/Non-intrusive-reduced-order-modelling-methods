#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:48 2021

@author: florianma

"""
from dolfin import (dx, solve, lhs, rhs, dot, assemble, grad, parameters)
prm = parameters['krylov_solver']  # short form


class ConvectionDiffusion():
    def __init__(self, domain, k=0.0, alpha=.0):
        """ k=1: fwd E
            k=0: bwd E
            k=.5: C-N"""
        D, dt = domain.D, domain.dt
        t, t_1, vt = domain.t, domain.t_1, domain.vt
        u_, boundary_terms = domain.u_, domain.robin_boundary_terms

        # Step 4: Transport of rho / Convection-diffusion and SUPG
        # vr = vr + tau_SUPG * inner(u_, grad(vr))  # SUPG stabilization
        # F4 = dot((t - t_1) / dt, vt)*dx + dot(div(t*u_), vt) * dx       + D*dot(grad(t), grad(vt)) * dx
        # above does not work, below works fine, but is mathematically not correct, since d/dt (rho) is not 0
        # F = dot((t - t_1) / dt, vt)*dx + dot(dot(grad(t), u_), vt) * dx \
        #     + D*dot(grad(t), grad(vt))*dx + boundary_terms
        # if alpha != 0.0:
        #     h = (domain.mesh.hmax()+domain.mesh.hmin())/2.
        #     mu = domain.get_mu().mean()
        #     mesh = domain.mesh
        #     w0 = domain.u_.compute_vertex_values(mesh)
        #     w0.shape = (2, -1)
        #     magnitude = np.linalg.norm(w0, axis=0).mean()
        #     Pe = magnitude * h / (2.0 * mu)
        #     tau = h / (2.0*magnitude) * (1.0/np.tanh(Pe) - 1.0/Pe)
        #     beta = Constant(tau*alpha)
        #     vt = vt + beta*h*grad(vt)
        t_mid = k*t + (1-k)*t_1
        F = (((t - t_1) / dt)*vt*dx + dot(u_, grad(t_mid))*vt*dx
             + D*dot(grad(t_mid), grad(vt))*dx + boundary_terms)
        self.F = F
        a, L = lhs(F), rhs(F)

        A = assemble(a)
        self.a, self.L, self.A = a, L, A
        self.domain = domain

        prm['absolute_tolerance'] = 1E-10
        prm['relative_tolerance'] = 1E-6
        prm['maximum_iterations'] = 1000
        return

    def solve(self):
        # solve(lhs(self.F) == rhs(self.F), self.domain.t_)
        bct, t_ = self.domain.bct, self.domain.t_
        A, L, a = self.A, self.L, self.a
        A = assemble(a)
        b = assemble(L)
        [bc.apply(A) for bc in bct]
        [bc.apply(b) for bc in bct]
        # bicgstab, might have stability issues
        # gmres: non-symmetric, take care of stopping parameters
        solve(A, t_.vector(), b, 'gmres', 'ilu')
        # parameters
        return
