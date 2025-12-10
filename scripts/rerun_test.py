import rerun as rr
import numpy as np
import rosbag
import trimesh
from scipy.spatial.transform import Rotation as R

rr.init("robot_trajectory_viewer", spawn=True)

bag_file = "../bags/gnm_bunker_mist_office_sharp_no_aug_trial_1.bag"
odom_topic = "/lvi_sam/lidar/mapping/odometry"
glb_file = "office_mesh_oak_first_corridor.glb"

# Load mesh
scene_or_mesh = trimesh.load(glb_file)
if isinstance(scene_or_mesh, trimesh.Scene):
    meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)
else:
    mesh = scene_or_mesh

vertices = np.array(mesh.vertices)
triangles = np.array(mesh.faces)
vertex_colors = np.ones((len(vertices), 3)) * 128

# ROTATION FIX:
# Option 2: Rotate 90 degrees around Y-axis
rotation = R.from_euler('y', 90, degrees=True).as_matrix()

# Apply rotation to vertices
vertices_rotated = (rotation @ vertices.T).T

# Now apply translation to align with first odom position
mesh_center_rotated = vertices_rotated.mean(axis=0)
first_odom_position = np.array([0.0, 0.0, 0.0])
offset = first_odom_position - mesh_center_rotated

vertices_aligned = vertices_rotated + offset

print(f"After rotation - mesh center: {mesh_center_rotated}")
print(f"After translation - mesh center: {vertices_aligned.mean(axis=0)}")
print(f"Mesh bounds X: [{vertices_aligned[:, 0].min():.2f}, {vertices_aligned[:, 0].max():.2f}]")
print(f"Mesh bounds Y: [{vertices_aligned[:, 1].min():.2f}, {vertices_aligned[:, 1].max():.2f}]")
print(f"Mesh bounds Z: [{vertices_aligned[:, 2].min():.2f}, {vertices_aligned[:, 2].max():.2f}]")

# Log aligned mesh
rr.log(
    "world/mesh",
    rr.Mesh3D(
        vertex_positions=vertices_aligned,
        triangle_indices=triangles,
        vertex_colors=vertex_colors
    ),
    static=True
)

# Visualize trajectory
bag = rosbag.Bag(bag_file)
trajectory_points = []

for _, msg, _ in bag.read_messages(topics=[odom_topic]):
    timestamp = msg.header.stamp.to_sec()
    rr.set_time_seconds("ros_time", timestamp)

    pos = msg.pose.pose.position
    position = [pos.x, pos.y, pos.z]

    ori = msg.pose.pose.orientation
    quaternion = [ori.x, ori.y, ori.z, ori.w]

    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=position,
            rotation=rr.Quaternion(xyzw=quaternion),
            axis_length=0.3
        )
    )

    trajectory_points.append(position)

    if len(trajectory_points) > 1:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([trajectory_points], colors=[0, 255, 0])
        )

bag.close()
print(f"\nLogged {len(trajectory_points)} trajectory points")
