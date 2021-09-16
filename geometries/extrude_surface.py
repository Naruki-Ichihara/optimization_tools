import trimesh as tr
import numpy as np

extrude_hight = 6.0
load_meshpath = 'step_implant_surface.obj'
export_meshpath = 'step_implant_extruded.obj'

mesh = tr.load_mesh(load_meshpath)
verts = np.array(mesh.vertices)
verts_2d = list(np.delete(verts, 2, 1))
extruded = tr.creation.extrude_triangulation(verts_2d, mesh.faces, extrude_hight)
extruded.export(export_meshpath)