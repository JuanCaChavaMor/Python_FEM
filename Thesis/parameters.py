# -*- coding: utf-8 -*-
"""
This gives a class for the parametersm of the ADR 
optimal control model problem.
"""

# Import modules
from dolfin import *
import matplotlib.pyplot as plt
from mshr import *

# Define class for the desired state
class DesiredState(UserExpression):
  def eval(self, values, x):
    values[0] = 18
    #values[0] = x[0] + x[1]
  def value_shape(self):
    return tuple()

##### Set data #####
initRef = 20	
mesh = UnitSquareMesh(initRef,initRef)
#plt.figure(1)
#plot(mesh)

# Define the problem class
class ParameterClass(object):

  # Constructor
  def __init__(self, debug=False, level=0, gamma=1e-3):

    # Store parameters passed at construction time
    self.debug = debug
    self.level = level
    self.gamma = gamma

    # Generate the geometry and its subdomains
    domain = Rectangle(dolfin.Point(0.0, 1.0), dolfin.Point(0.0, 1.0)) 
    control_domain = Rectangle(dolfin.Point(0.5, 0.5), dolfin.Point(2.0, 3.0)) \
        + Rectangle(dolfin.Point(3.0, 0.5), dolfin.Point(4.5, 3.0)) 
    observation_domain = Rectangle(dolfin.Point(0.2, 2.5), dolfin.Point(1.5, 3.3)) \
        + Polygon((dolfin.Point(3.5, 0.3), dolfin.Point(4.8, 0.3), dolfin.Point(4.8, 1.4)))
    # There is a third subdomain, the intersection of the control and observation subdomains
    control_observation_domain = control_domain * observation_domain

    # Assign subdomain numbers
    # Notice that the last subdomain defined at a point always takes precedence
    domain.set_subdomain(1, control_domain)
    domain.set_subdomain(2, observation_domain)
    domain.set_subdomain(3, control_observation_domain)

    # Generate the mesh
    self.mesh = generate_mesh(domain, 10*pow(2.0,self.level))

    # Set the subdomain marker mesh function
    self.subdomain_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), self.mesh.domains())

    # In debugging mode, plot the mesh and the subdomains
    if self.debug:
      plt.figure()
      plot(self.mesh, title="mesh of floor heating problem at level %d" %self.level)

      plt.figure()
      ps = plot(self.subdomain_marker)
      plt.colorbar(ps)
    
    # Setup the measure with subdomain markers for control and observation subdomains
    self.dx = Measure("dx")(subdomain_data=self.subdomain_marker)

    # Define the polynomial degrees for the control and the state discretization
    self.degree_control = 0
    self.degree_state = 1

    # Set up the function spaces for the control and the state
    # Note: We define the control over the entire mesh (instead of a SubMesh
    # containing only the control domain) due to difficulties of using function spaces
    # defined over different meshes within one UFL form
    self.U = FunctionSpace(self.mesh, "DG", self.degree_control)
    self.Y = FunctionSpace(self.mesh, "CG", self.degree_state)

    # Setup the desired state yd
    self.yd = Function(self.Y)
    self.yd.interpolate(DesiredState(degree=self.degree_state))

    # Setup the thermal conductivity coefficient
    self.kappa = 20

    # Setup the heat transfer coefficient
    self.alpha_wall = 1
    self.alpha_window = 3
    self.alpha = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
    self.alpha.set_all(1)
    Window().mark(self.alpha, 2)

    # Setup the boundary measure with subdomain markers for wall (0) and windows (1)
    self.ds = Measure("ds")(subdomain_data=self.alpha)

    # Evaluate the right hand side b
    self.b = self.project_onto_control_space(self.solve_backward_PDE(self.yd))

    # Evaluate the constant term in the objective c
    self.c = 0.5 * assemble(inner(self.yd,self.yd) * self.dx((2,3)))


  # Define the solver for the forward PDE 
  # \int_\Omega \nabla y \cdot \nabla v \dx + \int_\Gamma \alpha y v \ds
  #   = \int_\Omega_{ctrl} u v \dx
  def solve_forward_PDE(self, u):
  
    # Define trial and test functions
    y = TrialFunction(self.Y)
    v = TestFunction(self.Y)

    # Define the weak formulation 
    a = Constant(self.kappa) * inner(grad(y),grad(v)) * self.dx \
        + Constant(self.alpha_wall) * inner(y,v) * self.ds(1) \
        + Constant(self.alpha_window) * inner(y,v) * self.ds(2)
    L = inner(u,v) * self.dx((1,3))

    # Solve the forward PDE
    y = Function(self.Y)
    solve(a == L, y)

    # Return the state
    return y


  # Define the solver for the backward PDE 
  # \int_\Omega \nabla p \cdot \nabla v \dx + \int_\Gamma \alpha p v \ds
  #   = \int_\Omega_{obs} z v \dx
  def solve_backward_PDE(self, z):
  
    # Define trial and test functions
    p = TrialFunction(self.Y)
    v = TestFunction(self.Y)

    # Define the weak formulation 
    a = Constant(self.kappa) * inner(grad(p),grad(v)) * self.dx \
        + Constant(self.alpha_wall) * inner(p,v) * self.ds(1) \
        + Constant(self.alpha_window) * inner(p,v) * self.ds(2)
    L = inner(z,v) * self.dx((2,3))

    # Solve the backward PDE
    p = Function(self.Y)
    solve(a == L, p)

    # Return the state
    return p


  # L2-project a given function onto the control space
  def project_onto_control_space(self, p):

    # Define trial and test functions
    u = TrialFunction(self.U)
    v = TestFunction(self.U)

    # Define the weak formulation 
    a = inner(u,v) * self.dx
    L = inner(p,v) * self.dx((1,3))

    # Solve the forward PDE
    u = Function(self.U)
    solve(a == L, u)

    # Return the solution
    return u

  # Restrict the function to the control space on the relevant subdomain
  # for plotting purposes only
  def reduce_to_control_space(self, u):

    # Extract a submesh for the control domain
    combined_subdomains = MeshFunction("size_t", self.mesh, 2)
    combined_subdomains.array()[self.subdomain_marker.array() == 1] = 1
    combined_subdomains.array()[self.subdomain_marker.array() == 3] = 1
    submesh = SubMesh(self.mesh, combined_subdomains, 1)
    return project(u,FunctionSpace(submesh, "DG", self.degree_control))


  # Define the operator governing the optimal control problem
  def A(self, u): 

    # Solve the forward problem
    y = self.solve_forward_PDE(u)

    # Solve the backward problem
    p = self.solve_backward_PDE(y)

    # L2-project the solution onto the control space and add
    # gamma times input
    return self.project_onto_control_space(p) + self.gamma * u


  # Define the L2 inner product on the control domain
  # as the problem specific inner product
  def inner_product(self, u, v): 

    # Assemble the inner product of u and v
    return assemble(inner(u,v) * self.dx((1,3)))


# Main code to run when this file is run as a script
# ------------------------------------------------
if __name__ == "__main__": 

  # Set log level
  set_log_level(30)

  # Set debugging parameter
  debug = True
  
  # Set global parameters
  # parameters["allow_extrapolation"] = True

  # Setup an instance of the problem object
  fh = FloorHeatingClass(debug=debug, level=1)

  # Define a constant one control 
  u = fh.project_onto_control_space(Constant(10.0))

  # Solve the forward problem with this control
  y = fh.solve_forward_PDE(u)

  # Evaluate the operator A on this control
  fh.A(u)

  # Evaluate the right hand side b
  fh.b

  # Evaluate the constant in the objective c
  fh.c

  # Display the sample control and associated state
  if debug:
    plt.figure()
    pu = plot(u, title="Sample control")
    plt.colorbar(pu)

    plt.figure()
    pu = plot(fh.reduce_to_control_space(u), title="Sample control")
    plt.colorbar(pu)

    plt.figure()
    py = plot(y, title="Sample state")
    plt.colorbar(py)

    plt.figure()
    pyd = plot(fh.yd, title="Desired state")
    plt.colorbar(pyd)

    plt.figure()
    pb = plot(fh.b, title="RHS")
    plt.colorbar(pb)

    plt.show()

