# pip install trimesh pyglet

import trimesh
import numpy as np
import open3d as o3d

# ----------------------------
# 1. Load GLB
# ----------------------------

mesh_name = "office_mesh_downsample"

scene_or_mesh = trimesh.load(f"{mesh_name}.glb")

# If it's a Scene → merge all geometries into one mesh
if isinstance(scene_or_mesh, trimesh.Scene):
    meshes = [
        geom for geom in scene_or_mesh.geometry.values()
        if isinstance(geom, trimesh.Trimesh)
    ]
    mesh = trimesh.util.concatenate(meshes)
else:
    mesh = scene_or_mesh

# ----------------------------
# 2. Sample point cloud
# ----------------------------
N = 100_000
points, face_idx = mesh.sample(N, return_index=True)

# ----------------------------
# 3. Convert GLTF coordinates to Open3D coordinates
# The GLTF coordinate system is Y-up, while Open3D uses Z-up, so when visualizing the point cloud is upside down.
# GLTF: Y-up → Open3D: Z-up
# ----------------------------
T = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0]
])

points = points @ T.T


# Handle colors safely
if (mesh.visual is not None and
    hasattr(mesh.visual, "face_colors") and
    mesh.visual.face_colors is not None):

    colors = mesh.visual.face_colors[face_idx][:, :3] / 255.0
else:
    colors = None



# Top down view is obstructed by the ceiling points, so we can filter them out for better visualization
z_threshold = 2.5
mask = points[:, 2] < z_threshold
points_filtered = points[mask]
colors_filtered = colors[mask] # To keep rgb we also need to apply the mask to colors




# ----------------------------
# 3. Convert to Open3D
# ----------------------------
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points_filtered)

if colors is not None:
    cloud.colors = o3d.utility.Vector3dVector(colors_filtered)

# ----------------------------
# 4. Save + visualize
# ----------------------------
# o3d.io.write_point_cloud(f"cloud_{mesh_name}.ply", cloud)
# print(f"Saved cloud_{mesh_name}.ply")

o3d.visualization.draw_geometries([cloud])

