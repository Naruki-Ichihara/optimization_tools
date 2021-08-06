#!opt/conda/envs/project/bin/python
# -*- coding: utf-8 -*-
"""Topology and material orientation optimization tool set."""

from fenics import *
from fenics_adjoint import *
import numpy as np
from ufl import operators, transpose
from torch_fenics import *
import torch
import nlopt as nl

def iso_filter(z, e):
    """iso_filter

    Apply 2D isoparametric projection onto orientation vector.

    Args:
        z: 0-component of the orientation vector (on natural setting)
        e: 1-component of the orientation vector (on natural setting)

    Returns:
        [Nx, Ny] (fenics.vector): Orientation vector with unit circle boundary condition on real setting.
    """
    u = as_vector([-1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0, -1/np.sqrt(2), -1])
    v = as_vector([-1/np.sqrt(2), -1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0])
    N1 = -(1-z)*(1-e)*(1+z+e)/4
    N2 =  (1-z**2)*(1-e)/2
    N3 = -(1+z)*(1-e)*(1-z+e)/4
    N4 =  (1+z)*(1-e**2)/2
    N5 = -(1+z)*(1+e)*(1-z-e)/4
    N6 =  (1-z**2)*(1+e)/2
    N7 = -(1-z)*(1+e)*(1+z-e)/4
    N8 =  (1-z)*(1-e**2)/2
    N = as_vector([N1, N2, N3, N4, N5, N6, N7, N8])
    Nx = inner(u, N)
    Ny = inner(v, N)
    return as_vector((Nx, Ny))

def helmholtz_filter(u, U, r=0.025):
    """ helmholtz_filter

    Apply the helmholtz filter.

    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        r (float): filter radius

    Return:
        v (fenics.function): filtered function
    """
    v = TrialFunction(U)
    dv = TestFunction(U)
    vh = Function(U)
    a = r*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = dot(u, dv)*dx
    solve(a==L, vh)
    return project(vh, U)

def heviside_filter(u, U, offset=0.5):
    """heviside_filter
    
    Apply the heviside function

    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        offset (float): base line

    Returns:
        v (fenics.function): filterd function

    Note:
        TODO: Using Regularlized Heaviside function
    
    """
    return project((1-offset)*u/2 + (1+offset)/2, U)


class CoreProcess(torch_fenics.FEniCSModule):
    """Core class
    
    Apply some pre-process and eveluate the loss function.

    Attributes:
        mesh (fenics.Mesh): mesh for FE analysis.
        load_conditions (list[fenics.bc]):
        applied_load_vector (list[fenics.vector]):
        bcs (list[fenics.bc]): boundary condtions.
        files (dict{'str'}): File paths for saving some results.
    
    """
    def __init__(self, mesh, load_conditions, applied_load_vectors, bcs, files):
            self.mesh = mesh
            self.load_conditions = load_conditions
            self.applied_loadvectors = applied_load_vectors
            self.bcs = bcs
            self.files = files
    
    def input_templates(self):
        return Function(FunctionSpace(self.mesh, 'CG', 1)), Function(FunctionSpace(self.mesh, 'CG', 1)), Function(FunctionSpace(self.mesh, 'CG', 1))

    def solve(self, z, e, r):
        """solve

        calcuate strain energy from given design parameters.

        Args:
            z: 0-component of the orientation vector (on natural setting)
            e: 1-component of the orientation vector (on natural setting)
            r: Relatively density field

        Returns:
            f: Strain energy
        """

        facets = MeshFunction('size_t', self.mesh, 1)
        facets.set_all(0)
        for i in range(len(self.load_conditions)):
            self.load_conditions[i].mark(facets, i+1)
        ds = Measure('ds', subdomain_data=facets)

        Orient = VectorFunctionSpace(self.mesh, 'CG', 1)
        orient = helmholtz_filter(project(iso_filter(z, e), Orient), Orient)
        orient.rename('Orientation vector field', 'label')

        offset = 0.5
        Density = FunctionSpace(self.mesh, 'CG', 1)
        density = heviside_filter(helmholtz_filter(r, Density), Density, offset=offset)
        density.rename('Relatively density field', 'label')


        

