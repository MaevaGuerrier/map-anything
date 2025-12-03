import trimesh
import open3d as o3d
import numpy as np
import rosbag # pip install bagpy https://stackoverflow.com/questions/59794328/importing-rosbag-in-python-3

glb_file = "12_3_2025.glb"

scene = trimesh.Scene()

mesh = trimesh.load(glb_file)

if isinstance(mesh, trimesh.Scene):
    scene = mesh  # Keep as scene
else:
    scene = trimesh.Scene(mesh)



mesh_origin = mesh.centroid  # or use mesh.bounds[0] for corner
axis_mesh = trimesh.creation.axis(
    origin_size=0.1,      # Size of origin sphere
    axis_radius=0.02,     # Thickness of axes
    axis_length=2.0,      # Length of each axis
    transform=trimesh.transformations.translation_matrix(mesh_origin)
)
scene.add_geometry(axis_mesh, node_name="mesh_axes")




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
# rot_mat = np.array([
#     [0, -1, 0],
#     [1, 0, 0],
#     [0, 0, 1]
# ])
# poses = poses @ rot_mat.T



R_world_to_opencv = np.array([
    [ 0,  1,  0,  0],
    [ 0,  0, -1,  0],
    [ 1,  0,  0,  0],
    [ 0,  0,  0,  1]
])


positions_corrected = poses @ R_world_to_opencv[:3, :3].T


offset = [2.4, 0, -1.2]
# print(positions_corrected.shape) # (1588, 3)
positions_corrected[:, :] += offset

path = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
    vertices=positions_corrected
)

# Add trajectory to scene (as a path with color)
scene.add_geometry(path, node_name="trajectory")


trajectory_end = positions_corrected[-1]
axis_traj_end = trimesh.creation.axis(
    origin_size=0.15,
    axis_radius=0.04,
    axis_length=3.0,
    transform=trimesh.transformations.translation_matrix(trajectory_end)
)
scene.add_geometry(axis_traj_end, node_name="trajectory_end_axes")



transform_rotation = np.array([
    [1, 0, 0, 0],  # x red
    [0, 1, 0, 0],  # y green 
    [0, 0, 1, 0],  # z blue
    [0, 0, 0, 1]
])

scene.apply_transform(transform_rotation)


scene.show()
