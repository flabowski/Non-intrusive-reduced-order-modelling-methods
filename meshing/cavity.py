# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:36:53 2022

@author: florianma
"""
import pygmsh
from meshing.general import to_file


def mesh(lcar=0.02, L=1.0):
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
        fluid = geom.add_surface_loop(s)
        # Add physical markers to gmsh
        # Triangles:
        # - 0: Fluid
        # Lines:
        # - 1: bottom
        # - 2: right
        # - 3: top
        # - 4: left
        btm = [c[0]]
        rgt = [c[1]]
        top = [c[2]]
        lft = [c[3]]
        geom.add_physical(fluid, "0")
        geom.add_physical(btm, "1")
        geom.add_physical(rgt, "2")
        geom.add_physical(top, "3")
        geom.add_physical(lft, "4")
        # When using physical markers, unused points are placed last in the mesh
        msh = geom.generate_mesh(dim=2)
    return msh


def save_cavity_mesh(lcar=0.02):
    # used to be create_cavity_mesh
    msh = mesh(lcar)
    to_file(msh)


if __name__ == "__main__":
    msh = mesh()