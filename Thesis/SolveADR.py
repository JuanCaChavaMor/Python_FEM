from dolfin import *

import matplotlib.pyplot as plt
import sys
import numpy as np

def SolveADR(mesh,PU,PY,beta,phi,yd,g,lam):
    ### Parameters
    epsi =Constant(1.0)
    kappa=Constant(1.0)
    be=Constant((3.0, 4.0))
    ### BC parameter for mixed formulation
    Dbc= Constant(0.0)
    ##### function spaces:  mixed formulation #####
    U = FunctionSpace(mesh,PU)
    Y = FunctionSpace(mesh,PY)
    # (u,y,p) \in (U,Y,Y) but we need the element spaces
    X = FunctionSpace(mesh,MixedElement([PU,PY,PY]))
    ## Boundary Conditions for State and Dual eqs.
    bcy = DirichletBC(X.sub(1),Dbc,'on_boundary')       
    bcp = DirichletBC(X.sub(2),Dbc,'on_boundary')
    BC=[bcy, bcp]
    ##### solve the ADR equations #####
    u,y,p = TrialFunctions(X)
    w,v,q = TestFunctions(X)
    ## Bilinear Forms
    Co= (beta*p*w+u*w)*dx
    Ay= (inner(epsi*grad(y),grad(v))+(inner(be,grad(y)))*v+kappa*y*v)*dx
    Ap= (inner(epsi*grad(p),grad(q))-(inner(be,grad(p)))*q+kappa*p*q)*dx
    a = Co+Ay+Ap
    L = (inner(g,v)+yd*q+phi*w)*dx
    ## Assign the solution
    uh_yh_ph = Function(X)
    ## Solve
    solve(a == L,uh_yh_ph,BC)
    #### Split components of the solution #####
    uh = Function(U)
    yh = Function(Y)
    ph = Function(Y)
    assign(uh,uh_yh_ph.sub(0))
    assign(yh,uh_yh_ph.sub(1))
    assign(ph,uh_yh_ph.sub(2))

    return(uh,yh,ph)


