#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:02:38 2021

@author: juancarlos
"""
from dolfin import *
import numpy as np

def L2Projection(mesh, f, polydegree):
    V = FunctionSpace(mesh,'DG',polydegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    r = u*v*dx
    rhs = f*v*dx
    Pu = Function(V)
    solve(r == rhs, Pu)
    
    return(Pu)