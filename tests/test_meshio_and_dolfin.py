#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:39:21 2021

@author: florianma
"""

import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import matplotlib as mpl
from dolfin import (Function, DirichletBC, Expression, TestFunction,
                    TrialFunction, Mesh, FunctionSpace, Constant, Measure,
                    VectorFunctionSpace, XDMFFile, MeshValueCollection, cpp)
from dolfin import (VectorElement, FiniteElement, inner, grad, dx, div, solve,
                    lhs, rhs, split, project, dot)
from mpi4py import MPI
import meshio

# Write mesh to XDMF which we can easily read into dolfin
# if MPI.COMM_WORLD.rank == 0:
#     #  Pick one of ['abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml',
#     # 'exodus', 'flac3d', 'gmsh', 'gmsh22', 'h5m', 'hmf', 'mdpa', 'med',
#     # 'medit', 'nastran', 'neuroglancer', 'obj', 'off', 'permas', 'ply',
#     # 'stl', 'su2', 'svg', 'tecplot', 'tetgen', 'ugrid', 'vtk', 'vtu',
#     # 'wkt', 'xdmf'], xdmf fails

fmt = "xdmf"


def create_entity_mesh(mesh, cell_type, prune_z=False,
                       remove_unused_points=False):
    """
    Given a meshio mesh, extract mesh and physical markers for a given entity.
    We assume that all unused points are at the end of the mesh.points
    (this happens when we use physical markers with pygmsh)
    """
    cells = mesh.get_cells_type(cell_type)
    try:
        # If mesh created with gmsh API it is simple to extract entity data
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    except KeyError:
        # If mehs created with pygmsh, we need to parse through cell sets and sort the data
        cell_entities = []
        cell_data = []
        cell_sets = mesh.cell_sets_dict
        for marker, set in cell_sets.items():
            for type, entities in set.items():
                if type == cell_type:
                    cell_entities.append(entities)
                    cell_data.append(np.full(len(entities), int(marker)))
        cell_entities = np.hstack(cell_entities)
        sorted = np.argsort(cell_entities)
        cell_data = np.hstack(cell_data)[sorted]
    if remove_unused_points:
        num_vertices = len(np.unique(cells.reshape(-1)))
        # We assume that the mesh has been created with physical tags,
        # then unused points are at the end of the array
        points = mesh.points[:num_vertices]
    else:
        points = mesh.points

    # Create output mesh
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

lcar = .02
L = 1.0
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



input_mesh = create_entity_mesh(msh, "triangle", True, True)
meshio.write("mesh."+fmt, input_mesh, file_format=fmt)


mesh_name, facet_name = "mesh."+fmt, "mf."+fmt

mesh = Mesh()
with XDMFFile(mesh_name) as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh,
                          mesh.topology().dim() - 1)
with XDMFFile(facet_name) as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

