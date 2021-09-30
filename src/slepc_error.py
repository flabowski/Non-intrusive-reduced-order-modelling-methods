#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:52:44 2021

@author: florianma
"""
import sys
import slepc4py
slepc4py.init(sys.argv)


from slepc4py import SLEPc
from petsc4py import PETSc
Vs = SLEPc.BV().create(PETSc.COMM_SELF)
Vs.setSizes((30, 30), 30)
Vs.setActiveColumns(0, 30)
Vs.setFromOptions()
(r_loc, r_glob), cols = Vs.getSizes()
for i in range(cols):
    print(i)
    tmp = Vs.getColumn(i)
    print("got tmp")
    Vs.restoreColumn(i, tmp)
    print("restored tmp")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:52:44 2021

@author: florianma
"""
import sys
import slepc4py
slepc4py.init(sys.argv)


from slepc4py import SLEPc
from petsc4py import PETSc
Vs = SLEPc.BV().create(PETSc.COMM_SELF)
Vs.setSizes((30, 30), 30)
Vs.setActiveColumns(0, 30)
Vs.setFromOptions()
(r_loc, r_glob), cols = Vs.getSizes()
for i in range(cols):
    print(i)
    tmp = Vs.getColumn(i)
    print("got tmp")
    Vs.restoreColumn(i, tmp)
    print("restored tmp")
