#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 21:47:48 2021

@author: juancarlos
"""

from dolfin import *
import numpy as np

def gradedrefine(mesh):
	'''computes graded refinement
		Input: mesh
		Output: mesh
	'''   
	
	DG = FunctionSpace(mesh,'DG',0)
	dg0 = Function(DG)
	nrCells = dg0.vector()[:].size
	dg0.vector()[:] = range(0,nrCells)
	val = np.int(dg0(([.652,.511])))				
	dg0.vector()[:] = np.zeros(nrCells)
	cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
	cell_markers.set_all(False)
	cell = Cell(mesh,val)
	cell_markers[cell] = True
	mesh = refine(mesh,cell_markers)
	
	return(mesh)

def grefine(mesh):

	mesh = refine(mesh)							# uniform mesh refinement
	
	for j in range(0,3):
		mesh = gradedrefine(mesh)					# mesh refinement

	return(mesh)