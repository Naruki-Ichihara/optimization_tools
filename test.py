from Optimizer import Optimizer
from fenics import *
from fenics_adjoint import *
import numpy as np

problem = Optimizer()
mesh = Mesh('/workdir/mesh/implant.xml')
N = mesh.num_vertices()

class Clamp(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < -97.65 and on_boundary

class Loading(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 16.5 and on_boundary

clamp_bc = DirichletBC(VectorFunctionSpace(mesh, 'CG', 1), Constant((0, 0)), Clamp())
bcs = [clamp_bc]
boundaries = [Loading()]
applied_loads = [Constant((0, -1))]
material = {'E1': 3600, 'E2': 600, 'nu12':0.33, 'G12': 300}
path = 'results/'
files = {'Displacement': XDMFFile('{}displacement.xdmf'.format(path)),
         'Stress': XDMFFile('{}stress.xdmf'.format(path)),
         'Strain': XDMFFile('{}strain.xdmf'.format(path)),
         'Orient': XDMFFile('{}orient.xdmf'.format(path)),
         'Orientpipe': File('{}orient.xml'.format(path)),
         'Denspipe': File('{}dens.xml'.format(path)),
         'Dens': XDMFFile('{}dens.xdmf'.format(path))
        }
z0 = np.ones(N)
e0 = np.zeros(N)
r0 = np.ones(N)
x0 = np.concatenate([z0, e0, r0])
problem.set_mesh(mesh)
problem.set_bcs(bcs)
problem.set_loading(boundaries, applied_loads)
problem.set_material(material)
problem.set_working_dir(files)
problem.set_target(0.8)
problem.initialize()
problem.run(x0)

