from Optimizer.Optimizer import Optimizer
from fenics import *
from fenics_adjoint import *
import numpy as np

problem = Optimizer()
mesh = Mesh('/workdir/mesh/implant_step.xml')
N = mesh.num_vertices()

#mesh = RectangleMesh(Point(0,0), Point(40, 20), 100, 50)
#N = mesh.num_vertices()

class Clamp(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < -100.0 and on_boundary

class Loading(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 18.50 and on_boundary

displacement_boundaries_0 = [Clamp()]
displacement_boundaries_1 = [Clamp()]
applied_disp_0 = [Constant(0)]
applied_disp_1 = [Constant(0)]

loading_boundaries = [Loading()]
applied_loads = [Constant((0, -1))]

material = {'E1': 3600, 'E2': 600, 'nu12':0.45, 'G12': 500}
path = 'results/implant_SOLID/'
files = {'Displacement': XDMFFile('{}displacement.xdmf'.format(path)),
         'Stress': XDMFFile('{}stress.xdmf'.format(path)),
         'Strain': XDMFFile('{}strain.xdmf'.format(path)),
         'Orient': XDMFFile('{}orient.xdmf'.format(path)),
         'Orientpipe': File('{}orient.xml'.format(path)),
         'Denspipe': File('{}dens.xml'.format(path)),
         'Dens': XDMFFile('{}dens.xdmf'.format(path))
        }

z0 = np.zeros(N)
e0 = np.ones(N)
r0 = np.ones(N)
x0 = np.concatenate([z0, e0, r0])

problem.set_mesh(mesh)
problem.set_bcs(displacement_boundaries_0, displacement_boundaries_1, applied_disp_0, applied_disp_1)
problem.set_loading(loading_boundaries, applied_loads)
problem.set_material(material)
problem.set_working_dir(files)
problem.set_target(1.1)
problem.initialize()
problem.run(x0)

