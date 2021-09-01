# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:10:56 2021

@author: florianma
"""
import vtk
from vtk.util.numpy_support import vtk_to_numpy

filename = '/home/florianma@ad.ife.no/Documents/NIROM_Data/data/Foam_VTK_ParaMu/Test_010_1.vtk'
reader = vtk.vtkDataSetReader()
reader.SetFileName(filename)
reader.ReadAllScalarsOn()  # Activate the reading of all scalars
reader.Update()

h = reader.GetHeader()
print(h)

data = reader.GetOutput()


for sz in [data.GetNumberOfCells, data.GetNumberOfPolys,
           data.GetNumberOfLines, data.GetNumberOfStrips,
           data.GetNumberOfPieces, data.GetNumberOfVerts,
           data.GetNumberOfPoints]:
    print(sz())
for i in range(data.GetNumberOfCells()):
    if i < 3:
        p = data.GetCell(i)
        print(i, p)

for i in range(data.GetNumberOfPolys()):
    if i < 3:
        print(i)
        p = data.GetPoly(i)
        print(i, p)
for i in range(data.GetNumberOfPoints()):
    if i < 3:
        p = data.GetPoint(i)
        print(i, p)


cells = reader.GetOutput().GetPolys()
nCells = cells.GetNumberOfCells()
array = cells.GetData()
# This holds true if all polys are of the same kind, e.g. triangles.
# assert(array.GetNumberOfValues()%nCells==0)
# nCols = array.GetNumberOfValues()//nCells
# first row is always 3
numpy_cells = vtk_to_numpy(array).reshape((nCells, 4))[:, 1:]

# nodes_vtk_array = reader.GetOutput().GetPoints().GetData() # .GetValue(11248*3-1)
pts = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
cells = vtk_to_numpy(reader.GetOutput().GetPolys().GetData()).reshape(-1, 3)
p = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))
uv = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(1))


# data.GetVerts().GetData().GetValue(0) 
print(reader.GetNumberOfScalarsInFile())

# v = data.GetVerts()
# for i in range(v.GetNumberOfValues()):
#     v.GetData().GetValue(0)
# # data.

# # # Recorded script from Mayavi2
# # # from mayavi import mlab
# # %gui qt
# import mayavi
# from mayavi.api import Engine
# from mayavi.modules.surface import Surface
# from mayavi.modules.streamline import Streamline
# import numpy as np
# # from mayavi import mlab
# engine = Engine()
# engine.start()
# # -------------------------------------------
# scene = engine.new_scene()
# vtk_file_reader = engine.open(filename)
# # vtk_file_reader.name = 'VTK file (slidlid11_100x100.0004.vtk) (timeseries)'
# # vtk_file_reader.file_path = 'C:\\Users\\florianma\\Documents\\Alsim\\slidlid11_100x100.0004.vtk'
# vtk_file_reader.timestep = 4
# surface = Surface()
# engine.add_filter(surface, vtk_file_reader)
# module_manager = vtk_file_reader.children[0]
# module_manager.scalar_lut_manager.lut_mode = 'RdYlBu'
# streamline = Streamline()
# engine.add_filter(streamline, module_manager)
# streamline.actor.mapper.scalar_range = np.array([719.1506958, 799.99993896])
# streamline.actor.mapper.progress = 1.0
# streamline.actor.mapper.scalar_visibility = False
# # mlab.show()


# import numpy
# # from vtk import vtkStructuredPointsReader
# from vtk.util import numpy_support as VN

# reader = vtkStructuredPointsReader()
# reader.SetFileName(filename)
# reader.ReadAllVectorsOn()
# reader.ReadAllScalarsOn()
# reader.Update()

# data = reader.GetOutput()

# dim = data.GetDimensions()
# vec = list(dim)
# vec = [i-1 for i in dim]
# vec.append(3)

# u = VN.vtk_to_numpy(data.GetCellData().GetArray('velocity'))
# b = VN.vtk_to_numpy(data.GetCellData().GetArray('cell_centered_B'))

# u = u.reshape(vec,order='F')
# b = b.reshape(vec,order='F')

# x = zeros(data.GetNumberOfPoints())
# y = zeros(data.GetNumberOfPoints())
# z = zeros(data.GetNumberOfPoints())

# for i in range(data.GetNumberOfPoints()):
#         x[i],y[i],z[i] = data.GetPoint(i)

# x = x.reshape(dim,order='F')
# y = y.reshape(dim,order='F')
# z = z.reshape(dim,order='F')