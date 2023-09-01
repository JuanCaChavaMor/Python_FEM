#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:43:57 2021
This script creates the indicatrix function of the Active set A^a.
It needs three inputs: the functions to compare, f and ua as well as the 
function space U.
@author: juancarlos
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def project_up(f,U,ua):
    # Build the vectors to compare
    fv = f.vector()[:]
    upperv = ua.vector()[:]
    # Indicatrix function
    cond = np.where(fv<upperv,1,0)
    # Build the vector with the binary values
    proj_up = Function(U)
    proj_up.vector()[:] = cond

    return proj_up
