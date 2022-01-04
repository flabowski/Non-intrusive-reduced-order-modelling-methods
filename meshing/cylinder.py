# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:36:43 2022

@author: florianma
"""
import pygmsh
from meshing.general import to_file


def mesh(lcar):
    with pygmsh.geo.Geometry() as geom:
        r = .05
        p = [geom.add_point([.20, .20], lcar),
             geom.add_point([0.0, .0], lcar),
             geom.add_point([2.2, .0], lcar),
             geom.add_point([2.2, .41], lcar),
             geom.add_point([0.0, .41], lcar),
             geom.add_point([.2 + r, .20], lcar),
             geom.add_point([.20, .2 + r], lcar),
             geom.add_point([.2 - r, .20], lcar),
             geom.add_point([.20, .2 - r], lcar)]
        c = [geom.add_line(p[1], p[2]),  # btm
             geom.add_line(p[2], p[3]),
             geom.add_line(p[3], p[4]),  # top
             geom.add_line(p[4], p[1]),
             geom.add_circle_arc(p[5], p[0], p[6]),
             geom.add_circle_arc(p[6], p[0], p[7]),
             geom.add_circle_arc(p[7], p[0], p[8]),
             geom.add_circle_arc(p[8], p[0], p[5])]
        ll1 = geom.add_curve_loop([c[0], c[1], c[2], c[3]])
        ll2 = geom.add_curve_loop([c[4], c[5], c[6], c[7]])
        s = [geom.add_plane_surface(ll1, [ll2])]
        fluid = geom.add_surface_loop(s)

        # Add physical markers to gmsh
        # Triangles:
        # - 0: Fluid
        # Lines:
        # - 1: Top and bottom wall
        # - 2: Cylinder wall
        # - 3: Inlet
        # - 4: Outlet
        top_and_bottom = [c[0], c[2]]
        cylinderwall = c[4:]
        inlet = [c[3]]
        outlet = [c[1]]
        geom.add_physical(fluid, "0")
        geom.add_physical(top_and_bottom, "1")
        geom.add_physical(cylinderwall, "2")
        geom.add_physical(inlet, "3")
        geom.add_physical(outlet, "4")
        # When using physical markers, unused points are placed last in the mesh
        msh = geom.generate_mesh(dim=2)
    return msh


def save_channel_mesh(lcar=0.02):
    # used to be create_channel_mesh
    msh = mesh(lcar)
    to_file(msh)


if __name__ == "__main__":
    save_channel_mesh()