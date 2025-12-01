# pip install trimesh pyglet

import trimesh

# Path to your .glb file
mesh = trimesh.load("office_mesh_downsample.glb")

# Open an interactive 3D viewer window
mesh.show()
