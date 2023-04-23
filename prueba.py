# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:15:43 2020

@author: juancarlos
"""

from dolfin import *
from mshr import *


import matplotlib.pyplot as plt
import sys
import numpy as np


##### Set data #####
initRef = 36                                      				#initial refinement
nrRef = 6												#number of uniform mesh refinements
polydegree = 1		   								 	#polynomial degree k of the Lagrange space L^1_k(mathcal(T))
w = 1e-5


# Create mesh and define function space
#mesh = UnitSquareMesh(initRef, initRef)
square = Rectangle(Point(-1, -1), Point(1, 1))
#cutout = Rectangle(Point(+0, +0), Point(1, 1))
cutoutb = Polygon([Point(0,-w/2), Point(1,-w/2), Point(1,w/2), Point(0, w/2)])
domain = square - cutoutb
mesh   = generate_mesh(domain, initRef)

V = FunctionSpace(mesh, "Lagrange", polydegree)

# Define Dirichlet boundary 
def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(5.0)
a = inner(grad(u), grad(v))*dx
L = f*v*dx 

# Compute solution
uh = Function(V)
solve(a == L, uh, bc)
ndof = uh.vector().get_local().size
print("number of degrees of freedom =", ndof)
# Save solution in VTK format
file = File("poisson.pvd")
file << uh

plt.figure(1)
plot(uh, interactive=True, title="2(b). Poisson Solution for f=5, initRef=3, degree 2. Chavarría")
plt.savefig('solution_7_2.pdf')  
plt.figure(2)
plot(mesh, title="2(e). Bonus Mesh. Chavarría")
plt.savefig('bunus-mesh.pdf')  
# Plot solution
#plot(uh, interactive=True)