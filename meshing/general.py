# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:36:29 2022

@author: florianma
"""
import numpy as np
import meshio
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = False


def with_MPI(fnc):
    # Write mesh to XDMF which we can easily read into dolfin
    def wrapper(*args, **kwargs):
        if MPI:
            if MPI.COMM_WORLD.rank == 0:
                fnc(*args, **kwargs)
            MPI.COMM_WORLD.barrier()
        else:
            fnc(*args, **kwargs)
    return wrapper


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


@with_MPI  # it is just an easier way of saying to_file = with_MPI(to_file)
def to_file(msh):
    #  Pick one of ['abaqus', 'ansys', 'avsucd', 'cgns', 'dolfin-xml',
    # 'exodus', 'flac3d', 'gmsh', 'gmsh22', 'h5m', 'hmf', 'mdpa', 'med',
    # 'medit', 'nastran', 'neuroglancer', 'obj', 'off', 'permas', 'ply',
    # 'stl', 'su2', 'svg', 'tecplot', 'tetgen', 'ugrid', 'vtk', 'vtu',
    # 'wkt', 'xdmf'], xdmf fails sometimes
    input_mesh = create_entity_mesh(msh, "triangle", False, True)
    facet_mesh  =  create_entity_mesh(msh, "line", False)
    
    meshio.write("../mesh.xdmf", input_mesh, file_format="xdmf")
    meshio.write("../mf.xdmf", facet_mesh, file_format="xdmf")
    return

