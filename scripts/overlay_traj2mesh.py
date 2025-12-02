import trimesh
import open3d as o3d
import numpy as np
import rosbag # pip install bagpy https://stackoverflow.com/questions/59794328/importing-rosbag-in-python-3

# ============================================================
# 1. LOAD GLB AND MERGE SCENE
# ============================================================
glb_file = "office_mesh_oak_first_corridor.glb"

scene_or_mesh = trimesh.load(glb_file)

if isinstance(scene_or_mesh, trimesh.Scene):
    meshes = [
        g for g in scene_or_mesh.geometry.values()
        if isinstance(g, trimesh.Trimesh)
    ]
    mesh = trimesh.util.concatenate(meshes)
else:
    mesh = scene_or_mesh


# ============================================================
# 2. SAMPLE COLORED POINT CLOUD
# ============================================================
N = 150_000
points, face_idx = mesh.sample(N, return_index=True)

# Extract face colors if they exist
if mesh.visual and hasattr(mesh.visual, "face_colors"):
    colors = mesh.visual.face_colors[face_idx][:, :3] / 255.0
else:
    colors = np.zeros((len(points), 3))


# ============================================================
# 3. TRANSFORM COORDINATES (GLTF Y-up → Z-up)
# ============================================================
T_glb_to_o3d = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0]
])

points = points @ T_glb_to_o3d.T


# ============================================================
# 4. REMOVE CEILING (top 5%)
# ============================================================
# Top down view is obstructed by the ceiling points, so we can filter them out for better visualization
z_threshold = 1
mask = points[:, 2] < z_threshold
points_filtered = points[mask]
colors_filtered = colors[mask] # To keep rgb we also need to apply the mask to colors


# ============================================================
# 5. LOAD ROSBAG ODOMETRY
# ============================================================
bag = rosbag.Bag("../bags/gnm_bunker_mist_office_sharp_no_aug_trial_1.bag")

poses = []


for _, msg, _ in bag.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
    p = msg.pose.pose.position
    poses.append([p.x, p.y, p.z])

bag.close()

poses = np.array(poses)

# ROS typically uses: X-forward, Y-left, Z-up
# After mesh transformation, we have: X-right, Y-forward, Z-up
# So we may need to swap/flip axes to match

# Try this transformation (adjust based on your specific coordinate frames):
# Option 1: If ROS is X-forward, Y-left, Z-up

rot_mat = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
poses = poses @ rot_mat.T

positions_corrected = poses.copy()

offset = 2.5
positions_corrected[:, 1] += offset  # Shift Y on the viz Y is forward


segments = np.stack([positions_corrected[:-1], positions_corrected[1:]], axis=1)
path = trimesh.load_path(segments)


# ============================================================
# 8. COMBINE MESH + POINT CLOUD + PATH IN ONE VIEWER
# ============================================================
scene = trimesh.Scene()

# Debug: Print coordinate ranges
print("Mesh points range:")
print(f"  X: {points_filtered[:, 0].min():.2f} to {points_filtered[:, 0].max():.2f}")
print(f"  Y: {points_filtered[:, 1].min():.2f} to {points_filtered[:, 1].max():.2f}")
print(f"  Z: {points_filtered[:, 2].min():.2f} to {points_filtered[:, 2].max():.2f}")

print("\nOdometry range (after transform):")
print(f"  X: {positions_corrected[:, 0].min():.2f} to {positions_corrected[:, 0].max():.2f}")
print(f"  Y: {positions_corrected[:, 1].min():.2f} to {positions_corrected[:, 1].max():.2f}")
print(f"  Z: {positions_corrected[:, 2].min():.2f} to {positions_corrected[:, 2].max():.2f}")

# Add coordinate axes to scene for reference
axes = trimesh.creation.axis(origin_size=0.1, axis_length=2.0)
scene.add_geometry(axes)

# Add filtered point cloud
scene.add_geometry(trimesh.points.PointCloud(points_filtered, colors_filtered))

# Add odometry trajectory
scene.add_geometry(path)

scene.show()
