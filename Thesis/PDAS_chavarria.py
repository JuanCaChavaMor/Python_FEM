#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:47:55 2021

@author: Juan Carlos Chavarría

This script will return the solution of a Optimal Control Problem 
with restrictions given by an PDE and a box condition using the PDAS.
The script is general. The user provide the PDE, used as an input (solver),
and run this routine to obtain the solution as well as the desired state used
to compute the values of the objective functional J.

-- In this case we consider the ADR equation studied in the Bachellorarbeit
"""
from fenics import *
from math import sqrt, inf
from dolfin import *
from SolverADR import *                      # Solver of the ADR eq.
from project_up import *
from project_low import *
import numpy as np
from L2projection import *
import matplotlib.pyplot as plt
from plotslopes import *
#from mpltools import annotation
# Set some algorithmic parameters
atol = 1e-9
maxiter = 100
# Set the lambda parameter
lam = Constant(1.0)
# Define polynomial degrees for control and state discretization
degree_con = 0
degree_state = 1
# Exact Solutions
ye=Expression('x[0]*x[1]*(1-x[0])*(1-x[1])',degree=4)
pe=Expression('x[0]*x[1]*(1-x[0])*(1-x[1])',degree=4)
ue=Expression('min(2.0, max(-2.0, -((1-x[0])*x[0]*(1-x[1])*x[1])/lam))',lam=lam,degree=3)
#Define Mesh
# number of uniform refinements
Nref = 5  
### Store information ###
# a counter
array=np.arange(Nref)
# NDOF of each mesh
l=np.empty(Nref, dtype=int)
# Maximum diameter
h=np.empty(Nref)
# Errors
L2error = np.empty(Nref)
State_error_y = np.empty(Nref)
Adjointerror_p = np.empty(Nref)                             
### Loops ###
# exterior loop: set each mesh and calculate the approx. errors
for j in array:
    ## NDOF ##
    l[j]=2**(j+4)
    ## mesh size 
    h[j]=np.divide(1,(np.multiply(2,l[j])))  
    ## mesh                                     
    mesh = UnitSquareMesh(l[j],l[j])
    ##### Spaces ####
    ## Finite Element Spaces
    PU = FiniteElement('DG',mesh.ufl_cell(),degree_con)
    PY = FiniteElement('Lagrange',mesh.ufl_cell(),degree_state)
    ## Function Spaces
    U = FunctionSpace(mesh,PU)
    Y = FunctionSpace(mesh,PY)
    # Desired State 
    yom=Expression('(2*x[0] - 2*pow(x[0],2) + x[1] -14*x[0]*x[1] + 8*x[1]*pow(x[0],2) \
                  -pow(x[1],2) + 6*x[0]*pow(x[1],2)) ', degree=degree_state)
    yd=interpolate(yom,Y)
    # RHS state equation
    exf=Expression('(6*x[0]-6*pow(x[0],2)+5*x[1]-13*x[0]*x[1]\
              +7*pow(x[0],2)*x[1]-5*pow(x[1],2)+5*x[0]*pow(x[1],2)\
                  +pow(x[0],2)*pow(x[1],2)\
                      -min(2.0, max(-2.0, -(1-x[0])*x[0]*(1-x[1])*x[1])))'\
                          ,degree=degree_state)
    g=interpolate(exf,Y)
    # Set the bounds of box condition
    ua = interpolate(Expression('-2.0', degree=degree_con), U)  
    ub = interpolate(Expression('2.0', degree=degree_con), U)
    # Prepare for the PDAS loop
    # Set the initial guesses
    u0 = interpolate(Expression('-100.0', degree=degree_con), U)
    mu0 = interpolate(Expression('-100.0', degree=degree_con), U)
    # Create functions holding the iterations
    u = Function(U)
    mu= Function(U)
    toproj=Function(U)
    # # Build the constant 1 to use in beta function
    uno = interpolate(Expression('1.0', degree=degree_con), U)
    # Set the initial iterate
    u.assign(u0)
    mu.assign(mu0)
    toproj.assign(u0+mu0)
    ## Build the initial indicator functions
    Xa_old = project_up(toproj, U, ua)
    Xb_old = project_low(toproj, U, ub)
    # counter & condition
    iter = 0
    done = False
    # inner loop: PDAS
    while (iter < maxiter) and (done == False):
        beta = (uno-Xa_old-Xb_old)/lam  
        phi = ua*Xa_old+ub*Xb_old
        u, y, p = SolverADR(mesh, PU, PY, beta, phi, yd, g, lam)
        Ph=L2Projection(mesh, p, degree_con)
        tomu = np.add(np.multiply(1/lam,Ph.vector()[:]), np.multiply(-1, u.vector()[:]))
        mu.vector()[:] = tomu
        coefftoproj = np.add(u.vector()[:], mu.vector()[:])
        toproj.vector()[:] = coefftoproj
        Xa_new = project_up(toproj, U, ua)
        Xb_new = project_low(toproj, U, ub)
        error_a = sqrt(assemble((Xa_old-Xa_new)*(Xa_old-Xa_new)*dx))
        error_b = sqrt(assemble((Xb_old-Xb_new)*(Xb_old-Xb_new)*dx))
        print("Error is= ", error_a, error_b)
        if error_a+error_b < atol:
            done = True
        Xa_old = Xa_new
        Xb_old = Xb_new
        iter = iter+1
    # end of pdas loop
    print('number of iterations',iter)
    ## Errors: L2 for u, H1 for y and p
    L2error[j] = errornorm(ue,u,norm_type='L2')
    State_error_y[j]=errornorm(ye, y, norm_type='H1')
    Adjointerror_p[j]=errornorm(pe, p, norm_type='H1')
# end of loops
# print some information
print("el diámetro máximo de la malla es =", h )    
print("L2 error",L2error )
print("H1 error-state", State_error_y )
print("H1 error-adjoint", Adjointerror_p)
### Post Processing
# Plot the solutions
# and export to paraview
plt.figure(figsize=(15,5))
# Control
plt.subplot(1, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Control variable u')
vtkfile = File('control.pvd')
vtkfile << u
# State
plt.subplot(1,3,2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('State variable y')
vtkfile = File('state.pvd')
vtkfile << y
# Adjoint State
plt.subplot(1,3,3)
fig = plot(p)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title('Adjoint state p')
vtkfile = File('adjoint.pvd')
vtkfile << p
# Saving
plt.savefig('solution.pdf') 

# error plots
plt.figure(2)
plt.rcParams.update({'font.size': 8.5})
# convergence rate (adapted by hand)
# adjusted control
const_shift = -1.13571                    # shift of line in y-direction
const_rate = 1.00838                        # slope for line in loglog Plot
rate = pow(10,const_shift)*np.exp((const_rate*np.log(h)))    # function to plot
p1,=plt.loglog(h,rate,'m-.', label='Adjusted control')

# adjusted state
const_shift = -0.314198                   # shift of line in y-direction
const_rate = 0.99923                        # slope for line in loglog Plot
ratey = pow(10,const_shift)*np.exp((const_rate*np.log(h)))    # function to plot
p2,=plt.loglog(h,ratey,'r-.', label='Adjusted state')

# adjusted adjoint state
const_shift = -0.314022                   # shift of line in y-direction
const_rate = 0.999315                       # slope for line in loglog Plot
ratep = pow(10,const_shift)*np.exp((const_rate*np.log(h)))    # function to plot
p3,=plt.loglog(h,ratep,'g-.', label='Adjusted adj. state')
p4,=plt.loglog(h, L2error, 'y--o', label=r'$\Vert u-u_h\Vert_{L^2(\Omega)}$')
p5,=plt.loglog(h, State_error_y, '--D', label=r'$\Vert y-y_h\Vert_{H^1(\Omega)}$')
p6,=plt.loglog(h, Adjointerror_p, '--x', label=r'$\Vert p-p_h\Vert_{H^1(\Omega)}$')
slope_marker((h[2],L2error[2]-0.1*L2error[2]), (1,1))
#slope_marker((h[1],State_error_y[1]-0.1*State_error_y[1]), (1,1))
plt.xlabel('h')
plt.ylabel('Error')
plt.title("Convergence Rates, control, state and adjoint state")

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('convergence_ich.pdf') 
plt.show()
