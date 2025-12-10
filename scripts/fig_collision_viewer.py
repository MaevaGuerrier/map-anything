# pip install trimesh pyglet
import argparse
import trimesh
from trimesh.path.entities import Line


bunker = "bunker.glb"

# Path to your .glb file
obj = trimesh.load(bunker)

bbox = obj.bounding_box_oriented
corners = bbox.vertices  # 8 corner points


import numpy as np

# Edges of a cube (12 edges)
edges = np.array([
    [0,1], [1,2], [2,3], [3,0],   # bottom
    [4,5], [5,6], [6,7], [7,4],   # top
    [0,4], [1,5], [2,6], [3,7]    # vertical
])


entities = [Line(e) for e in edges]

red = [255, 0, 0, 255]

bbox_path = trimesh.path.Path3D(
    vertices=corners,
    entities=entities,
    colors=[red] * len(entities)
)


scene = trimesh.Scene()
scene.add_geometry(obj)
scene.add_geometry(bbox_path)

scene.show()
