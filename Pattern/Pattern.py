#!opt/conda/envs/project/bin/python
# -*- coding: utf-8 -*-
"""Stripe pattern"""

from functools import total_ordering
import numpy as np
from fenics import *
from mpi4py import MPI
comm = MPI.COMM_WORLD

class MeanFlow(UserExpression):
    def eval(self, val, x):
        val[0] = 1.0
        val[1] = 0.0
    def value_shape(self):
        return (2,)

class InitialConditions(UserExpression):
    def eval(self, val, x):
        val[0] = 0.54*np.random.randn()
        val[1] = 0.0
    def value_shape(self):
        return (2,)

class SwiftHohenbergEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

class KrylovSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, comm, PETScKrylovSolver(), PETScFactory.instance())
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")
        self.linear_solver().set_from_options()

class Terms():
    def __init__(self, epsilon, direction, frequency):
        self.q = frequency
        self.direction = direction
        self.eps = epsilon

    def term_A(self, w, v):
        return (dot(grad(w), grad(v)) - (self.q**2*w*v))*dx

    def term_A_aniso(self, w, v):
        u = self.direction
        D = outer(u, u)
        return (dot(grad(w), dot(D, grad(v))))*dx
    
    def Phi(self, w):
        return -self.eps*w**2/2 + w**4/4

    def Phi_p(self, w):
        return -self.eps*w + w**3

    def G1(self, w, v):
        return -self.eps/2 + (w**2 + w*v + v**2)/4

    def G2(self, v):
        return -self.eps*v/2 + v**3/4

def swift_hohenberg():
    '''compiler options'''
    # We use the Newton-Krylov method
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters['allow_extrapolation'] = True
    solver = KrylovSolver()

    mesh = Mesh(comm,'/workdir/mesh/implant.xml')
    orientation_path = '/workdir/garally/20210813_implant_80_30/morphogen/orient.xml'
    density_path = '/workdir/garally/20210813_implant_80_30/morphogen/dens.xml'
    output_path = '/workdir/garally/20210813_implant_80_30/pattern'
    ''''
    nx = 100
    ny = 100
    point_start = Point(0., 0.)
    point_end = Point(100, 100)
    mesh = RectangleMesh(comm, point_start, point_end, nx, ny)
    '''

    V = FiniteElement('CG', mesh.ufl_cell(), 2)
    ME = FunctionSpace(mesh, V*V)
    Refine_V = FunctionSpace(refine(mesh), 'CG', 1)
    Orientation_Space = VectorFunctionSpace(mesh, 'CG', 1)
    Density_space = FunctionSpace(mesh, 'CG', 1)

    density = Function(Density_space, density_path)
    orientation = Function(Orientation_Space, orientation_path)
    width = 0.8 / density
    frequency = np.pi/width

    U = TrialFunction(ME)
    phi, psi = TestFunctions(ME)
    Uh = Function(ME)
    Uh_0 = Function(ME)
    uh, qh = split(Uh)
    uh_0, qh_0 = split(Uh_0)
    U_init = InitialConditions()
    Uh.interpolate(U_init)
    Uh_0.interpolate(U_init)    

    elapsed = 0.0
    total_time = 50
    dt = 0.01
    mobility = 1.0
    file = XDMFFile(output_path + '/field.xdmf')
    lyapnov_path = output_path + '/lyapnpv.csv'
    lyapnov_list = []

    terms = Terms(5.0, orientation, frequency)

    verbosity = range(0, int(total_time//dt), 10)
    index = 0
    while (elapsed < total_time):
        qh_mid = 0.5*qh + 0.5*qh_0
        dPhi = terms.G1(uh, uh_0)*uh + terms.G2(uh_0)
        F1 = mobility*((uh-uh_0)*phi*dx + dt*terms.term_A(qh_mid, phi) + dt*dPhi*phi*dx + dt*terms.term_A_aniso(phi, uh))
        F2 = qh*psi*dx - terms.term_A(uh, psi)    
        F = F1 + F2
        a = derivative(F, Uh, U)
        problem = SwiftHohenbergEquation(a, F)
        elapsed += dt
        solver.solve(problem, Uh.vector())
        uh_solved = Uh.split()[0]
        k = frequency
        L = (0.5*(div(grad(uh_solved))**2 - 2*k**2*(dot(grad(uh_solved), grad(uh_solved)))\
             + k**4*uh_solved**2) + terms.Phi(uh_solved) + 0.5*dot(grad(uh_solved), dot(outer(orientation, orientation), grad(uh_solved))))*dx
        lyapnov_list.append(assemble(L))
        Uh_0.vector()[:] = Uh.vector()
        if index in list(verbosity):
            refine_uh_solved = interpolate(uh_solved, Refine_V)
            refine_uh_solved.rename('OrderParameter', 'label')
            file.write(refine_uh_solved, elapsed)
        index += 1
        np.savetxt(lyapnov_path, lyapnov_list, delimiter=',')
    pass

if __name__ == '__main__':
    swift_hohenberg()