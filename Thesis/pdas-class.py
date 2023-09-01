# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 23:33:09 2021

@author: juancarlos
"""

# Import modules
from math import sqrt, inf
from dolfin import Function, TestFunction
from SubspaceConjugateGradient import *
import numpy as np

# Define the PDAS solver class
class PDAS(object):

  # Constructor
  def __init__(self, problem, u0, mu0, ua=None, ub=None, lam=1, atol=1e-6, maxiter=100, debug=False):

    # Set parameters
    self.atol = atol
    self.maxiter = maxiter

    # Set the problem
    self.problem = problem

    # Set the bounds
    if ua is None:
      self.ua = np.full(u0.vector().size(), -inf)
    else:
      self.ua = ua

    if ub is None:
      self.ub = np.full(u0.vector().size(), inf)
    else:
      self.ub = ub

    # Set the PDAS parameter
    self.lam = lam

    # Set the initial guessesSplus
    self.u0 = u0
    self.mu0 = mu0

  # Define the solve function
  def solve(self):
# Create functions holding the iterates
    u = Function(self.problem.U)
    mu = Function(self.problem.U)
    res = Function(self.problem.U)

    # Set the initial iterate
    u.assign(self.u0)
    mu.assign(self.mu0)

    # Prepare for the PDAS loop
    iter = 0
    done = False

    # Enter the PDAS loop
    while not done:
      # Find the inner product of the Lagrange multiplier mu
      # with all control space basis functions
      v = TestFunction(self.problem.U)
      mu_tested = self.problem.inner_product(mu,v)

      # Determine the active sets
      Aa = mu_tested[:] + self.c * (u.vector()[:] - self.ua) < 0
      Ab  = mu_tested[:] + self.c * (u.vector()[:] - self.ub) > 0

      # Evaluate the residual of the complementarity condition 
      res.vector()[:] = np.where(Aa, u.vector()[:] - self.ua, -mu_tested)
      res.vector()[:] = np.where(Ab, u.vector()[:] - self.ub, s.vector()[:])

      # Evaluate the residual norm
      norm_res = sqrt(self.problem.inner_product(res,res))
      
      # Output some information
      print('----------------------------------------------')
      print(' ITER    |A-|    |A+|       |s|_U         TOL')
      print('----------------------------------------------')
      print(' %4d  %6d  %6d  %10.4e  %10.4e' % (iter, np.count_nonzero(Aa == True), np.count_nonzero(Ab == True), norm_res, self.atol))

      # Convergence (or maximum # of iterations reached) check
      if (norm_res <= self.atol) and (iter > 0):
        done = True
        flag = 0
      elif (iter == self.maxiter):
        done = True
        flag = 1

      if not done:

        # Prepare an initial guess for the subspace conjugate gradient solver
        # equal to the bounds on the active sets
        u0 = Function(self.problem.U)
        u0.vector()[:] = np.where(Aa, self.ua, u.vector()[:])
        u0.vector()[:] = np.where(Ab, self.ub, u0.vector()[:])

        # Solve the restricted problem on the current inactive set
        ix_fixed = Aa | Ab
        u, cg_iter, cg_flag = SubspaceConjugateGradient(problem=self.problem, u0=u0, ix_fixed=ix_fixed, rtol=1e-6, atol=1e-6, maxiter=10).solve()

        # Set the Lagrange multiplier
        mu.assign(self.problem.b - self.problem.A(u))

        # Increase counter
        iter = iter + 1

    # Return the final iterate and multiplier
    return u, mu, iter, flag