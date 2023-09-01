from dolfin import *

import matplotlib.pyplot as plt
import sys
import numpy as np

class DesiredState(UserExpression):
    def eval(self, values, x):
#    values[0]
    
#yd    = Expression('(2*x[0] - 2*pow(x[0],2) + x[1] -14*x[0]*x[1] + 8*x[1]*pow(x[0],2)-pow(x[1],2) + 6*x[0]*pow(x[1],2))',degree=3)
##### Set data #####
initRef = 20											#initial refinement\

##### define rhs, bc and mesh  #####
g =  Constant(1.0)
Dbc= Constant(0.0)

mesh = UnitSquareMesh(initRef,initRef)
#plt.figure(1)
#plot(mesh)

# Define the polynomial degrees for the control and the state discretization
degree_control = 1
degree_state = 1

    # Set up the function spaces for the control and the state
    # Note: We define the control over the entire mesh (instead of a SubMesh
    # containing only the control domain) due to difficulties of using function spaces
    # defined over different meshes within one UFL form
#    U = FunctionSpace(mesh, "DG", degree_control)
#    Y = FunctionSpace(mesh, "CG", degree_state)



##### function spaces:  mixed formulation #####
#PV = VectorElement('P',mesh.ufl_cell(),2)
#PQ = FiniteElement('P',mesh.ufl_cell(),1)
#Const = FiniteElement('R', mesh.ufl_cell(),0)

PU = FiniteElement('P',mesh.ufl_cell(),degree_control)
PY = FiniteElement('P',mesh.ufl_cell(),degree_state)



U = FunctionSpace(mesh,PU)
Y = FunctionSpace(mesh,PY)
# (u,y,p) \in (U,Y,Y) but we need the element spaces
X = FunctionSpace(mesh,MixedElement([PU,PY,PY]))

# Setup the desired state yd
#self.yd = Function(self.Y)
#self.yd.interpolate(DesiredState(degree=self.degree_state))

bcy = DirichletBC(X.sub(1),Dbc,'on_boundary')       # Check vector valued  BC
bcp = DirichletBC(X.sub(2),Dbc,'on_boundary')
BC=[bcy, bcp]
##### solve the Stokes equations #####
u,y,p = TrialFunctions(X)
w,v,q = TestFunctions(X)

Bv = (inner(u,w)+inner(grad(y),grad(v))+inner(grad(p),grad(q)))*dx
L = inner(g,v)*dx

uh_yh_ph = Function(X)
solve(Bv == L,uh_yh_ph,BC)

#### assign components of the solution #####
uh = Function(U)
yh = Function(Y)
ph = Function(Y)
assign(uh,uh_yh_ph.sub(0))
assign(yh,uh_yh_ph.sub(1))
assign(ph,uh_yh_ph.sub(2))

##### plot solution #####
plt.figure(2)
plot(uh)

plt.figure(3)
plot(yh)

plt.show()

