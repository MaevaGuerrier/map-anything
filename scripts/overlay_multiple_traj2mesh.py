from email.mime import image
import trimesh
from trimesh.creation import icosphere
import open3d as o3d
import numpy as np
import rosbag # pip install bagpy https://stackoverflow.com/questions/59794328/importing-rosbag-in-python-3
import argparse
from PIL import Image
import trimesh
import numpy as np
from trimesh.path.entities import Line
import trimesh.transformations as tf

def fig_bunker_collision():
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


    return obj, bbox_path


parser = argparse.ArgumentParser(description="3D GLB Viewer")
parser.add_argument(
    "--file",
    type=str,
    default="office.glb",
    help="Path to the .glb file to view",
)

parser.add_argument(
    "--env",
    type=str,
    default="office_loop",
)

parser.add_argument(
    "--robot",
    type=str,
    default="bunker"
)

parser.add_argument(
    "--aug",
    type=str,
    default="no_aug"
)

parser.add_argument(
    "--algo",
    type=str,
    default="bridger"
)

parser.add_argument(
    "--ref",
    type=str,
    default="reference_bunker_office_loop_reference_trial_1.bag",
    help="Name of the reference .bag file to read. Bag files are expected to be in ../bags/",
)

parser.add_argument(
    "--trial",
    type=int,
    default=1,
    help="Trial number to process",
)

parser.add_argument(
    "--col",
    action="store_true",
    help="Mark collision points in the trajectory",
)


args = parser.parse_args()


glb_file = args.file
ref_bag_file = args.ref
collision = args.col
env = args.env
robot = args.robot
aug = args.aug
algo = args.algo


scene = trimesh.Scene()

mesh = trimesh.load(glb_file)

if isinstance(mesh, trimesh.Scene):
    scene = mesh  # Keep as scene
else:
    scene = trimesh.Scene(mesh)


# Uncomment to add axes at mesh origin
# mesh_origin = mesh.centroid  # or use mesh.bounds[0] for corner
# axis_mesh = trimesh.creation.axis(
#     origin_size=0.1,      # Size of origin sphere
#     axis_radius=0.02,     # Thickness of axes
#     axis_length=2.0,      # Length of each axis
#     transform=trimesh.transformations.translation_matrix(mesh_origin)
# )
# scene.add_geometry(axis_mesh, node_name="mesh_axes")



if env == "office_loop":
    R_world_to_opencv = np.array([
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])
if env == "out_snow":
    R_world_to_opencv = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, 1,  0],
        [ 0,  -1,  0,  0],
        [ 0,  0,  0,  1]
    ])
if env == "hall":
    R_world_to_opencv = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, 1,  0],
        [ 0,  -1,  0,  0],
        [ 0,  0,  0,  1]
    ])


# OFFSET FOR OFFICE LOOP ENVIRONMENT
if env == "office_loop":
    offset_reference_trajectory = [2.5, -0.5, -1.45]
    if "bridger" in algo:
        offset = [2.5, -0.5, -1.45] #navibridger
    elif "gnm" in algo:
        offset = [2.5, -0.5, -1.45] # gnm
    elif "vint" in algo:
        offset = [2.5, -0.5, -1.45] # vint
    elif "nomad" in algo:
        offset = [2.70, -0.5, -1.40] # nomad
    else:
        offset = [2.5, -0.5, -1.45] # third value is too offset in z to align start position with robot position in mesh


# SNOW
if env == "out_snow":
    offset_reference_trajectory = [-9.0, -0.5, -1.0] #[-9.0, 0.0, -1.0]
    if "bridger" in algo:
        offset =[-9.0, -0.5, -1.0] #navibridger
    elif "gnm" in algo:
        offset = [-9.0, -0.5, -1.0]#= [-5.0, -0.5, -0.5] # gnm
    elif "vint" in algo:
        offset = [-9.0, -0.5, -1.0] # vint
    elif "nomad" in algo:
        offset = [-9.0, -0.5, -1.0] # nomad
    else:
        offset = [-9.0, -0.5, -1.0] # third value is too offset in z to align start position with robot position in mesh


if env == "hall":
    offset_reference_trajectory = [-9.0, -0.5, -1.0] #[-9.0, 0.0, -1.0]
    if "bridger" in algo:
        offset =[-9.0, -0.5, -1.0] #navibridger
    elif "gnm" in algo:
        offset = [-9.0, -0.5, -1.0]#= [-5.0, -0.5, -0.5] # gnm
    elif "vint" in algo:
        offset = [-9.0, -0.5, -1.0] # vint
    elif "nomad" in algo:
        offset = [-9.0, -0.5, -1.0] # nomad
    else:
        offset = [-9.0, -0.5, -1.0] # third value is too offset in z to align start position with robot position in mesh





if "bridger" in algo:
    algo_color = [255, 0, 0, 255] # red
elif "gnm" in algo:
    algo_color = [0, 0, 255, 255] # blue
elif "vint" in algo:
    algo_color = [255, 165, 0, 255] # orange
elif "nomad" in algo:
    algo_color = [255, 192, 255, 255] # pink


# Processing ALL trajectories



for trial in range (1, args.trial + 1):
    bag = rosbag.Bag(f"../bags/{algo}_{robot}_{env}_{aug}_trial_{trial}.bag")
    # reset
    poses = []
    positions_corrected = []
    path = None

    for _, msg, _ in bag.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
        p = msg.pose.pose.position
        poses.append([p.x, p.y, 0.0])

    bag.close()

    poses = np.array(poses)

    positions_corrected = poses @ R_world_to_opencv[:3, :3].T


    # third value is too offset in z to align start position with robot position in mesh
    # print(positions_corrected.shape) # (1588, 3)
    positions_corrected[:, :] += offset

    path = trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
        vertices=positions_corrected,
        colors=[algo_color]
    )
    scene.add_geometry(path, node_name=f"trajectory_trial_{trial}")


# Processing reference trajectory

bag_ref = rosbag.Bag(f"../bags/{ref_bag_file}")

poses_ref = []

# Processing reference trajectory
for _, msg, _ in bag_ref.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
    p = msg.pose.pose.position
    poses_ref.append([p.x, p.y, 0.0])

bag_ref.close()

poses_ref = np.array(poses_ref)

# ROS typically uses: X-forward, Y-left, Z-up
# After mesh transformation, we have: X-right, Y-forward, Z-up
# So we may need to swap/flip axes to match
# rot_mat = np.array([
#     [0, -1, 0],
#     [1, 0, 0],
#     [0, 0, 1]
# ])
# poses = poses @ rot_mat.T






positions_ref_corrected = poses_ref @ R_world_to_opencv[:3, :3].T



positions_ref_corrected[:, :] += offset_reference_trajectory



if collision:
    target_pos = positions_corrected[-1]
    obj, bbox_path = fig_bunker_collision()
    # reducing size
    scale_factor = 0.6   # shrink the object to 60%

    scale = tf.scale_matrix(scale_factor)

    obj.apply_transform(scale)
    bbox_path.apply_transform(scale)

    obj.apply_translation(target_pos)
    bbox_path.apply_translation(target_pos)

    scene.add_geometry(obj, node_name="bunker_collision")
    scene.add_geometry(bbox_path, node_name="bunker_collision_bbox")




# reference
path_ref = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_ref_corrected)))],
    vertices=positions_ref_corrected,
    colors=[[0, 255, 0, 255]]  # green for reference
)

# Add reference trajectory to scene
scene.add_geometry(path_ref, node_name="reference_trajectory")


# Uncomment to add axes
# axis_traj_end = trimesh.creation.axis(
#     origin_size=0.15,
#     axis_radius=0.04,
#     axis_length=3.0,
#     transform=trimesh.transformations.translation_matrix(positions_corrected[0])
# )
# scene.add_geometry(axis_traj_end, node_name="trajectory_end_axes")



# transform_rotation = np.array([
#     [1, 0, 0, 0],  # x red
#     [0, 1, 0, 0],  # y green
#     [0, 0, 1, 0],  # z blue
#     [0, 0, 0, 1]
# ])

# scene.apply_transform(transform_rotation)


scene.show()
