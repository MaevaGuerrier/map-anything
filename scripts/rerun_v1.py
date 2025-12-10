import rerun as rr
import numpy as np
import rosbag

import trimesh

rr.init("robot_traj_viewer", spawn=True)

trajectory_points = []

bag_file = "../bags/gnm_bunker_mist_office_sharp_no_aug_trial_1.bag"
odom_topic = "/lvi_sam/lidar/mapping/odometry"

glb_file = "office_mesh_oak_first_corridor.glb"

# mesh processing

scene_or_mesh = trimesh.load(glb_file)

if isinstance(scene_or_mesh, trimesh.Scene):
    meshes = [
        g for g in scene_or_mesh.geometry.values()
        if isinstance(g, trimesh.Trimesh)
    ]
    mesh = trimesh.util.concatenate(meshes)
else:
    mesh = scene_or_mesh


# if mesh.visual and hasattr(mesh.visual, "face_colors"):
#     vertex_colors = mesh.visual.face_colors[face_idx][:, :3] / 255.0
# else:
#     vertex_colors = np.zeros((len(vertices), 3))


vertices = np.array(mesh.vertices)
triangles = np.array(mesh.faces)

vertex_colors = np.ones((len(vertices), 3)) * 128

T_glb_to_o3d = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0]
])

vertices = (T_glb_to_o3d @ vertices.T).T


rr.log(
    "world/mesh",
    rr.Mesh3D(
        vertex_positions=vertices,      # (N, 3) array of vertex positions
        triangle_indices=triangles,     # (M, 3) array of triangle connectivity
        vertex_colors=vertex_colors     # (N, 3) array of RGB colors per vertex
    ),
    static=True  # Mesh doesn't change over time
)

# bag processing

bag = rosbag.Bag(bag_file)

for _, msg, _ in bag.read_messages(topics=[odom_topic]):

    timestamp = msg.header.stamp.to_sec()
    # Rest of the code same as before
    time_sec = timestamp
    rr.set_time_seconds("ros_time", time_sec)

    pos = msg.pose.pose.position
    position = [pos.x, pos.y, pos.z]

    ori = msg.pose.pose.orientation
    quaternion = [ori.x, ori.y, ori.z, ori.w]

    rr.log(
        "world/robot",
        rr.Transform3D(
            translation=position,
            rotation=rr.Quaternion(xyzw=quaternion)
        )
    )

    trajectory_points.append(position)

    if len(trajectory_points) > 1:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([trajectory_points], colors=[0, 255, 0])
        )
