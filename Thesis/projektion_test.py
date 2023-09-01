#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:43:57 2021

@author: juancarlos
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

M=100
mesh = UnitSquareMesh(M, M)
f = Expression('sin(x[0]*x[1])', degree=1)
V = FunctionSpace(mesh, "DG", 0)

fint = interpolate(f, V)
upper = interpolate(Expression('0.5',degree=1), V)

fv = fint.vector()[:]
upperv = upper.vector()[:]
cond = np.where(fv<upperv,1,0)

proj = Function(V)
proj.vector()[:] = cond

plt.figure(1)
c0 = plot(fint, mesh, title="Given Function")
plt.colorbar(c0)

plt.figure(2)
c1 = plot(proj, mesh, title="P1-Indicator Function")
plt.colorbar(c1)
plt.show()