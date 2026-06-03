import argparse
import itertools

import matplotlib.colors as mcolors
import numpy as np
import rosbag  # pip install bagpy https://stackoverflow.com/questions/59794328/importing-rosbag-in-python-3
import trimesh


def rotate_trajectory(positions, angle_deg, axis="z"):
    """
    Rotate trajectory around specified axis.

    Args:
        positions: Nx3 array of positions
        angle_deg: rotation angle in degrees
        axis: 'x', 'y', or 'z'
    """
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    if axis == "z":
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    elif axis == "y":
        rotation_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    elif axis == "x":
        rotation_matrix = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return (rotation_matrix @ positions.T).T


def generate_all_axis_aligned_transforms():
    transforms = []

    # All permutations of axes
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1

        # All sign combinations
        for signs in itertools.product([-1, 1], repeat=3):
            S = np.diag(signs)

            R = S @ P  # rotation/reflection matrix

            # Embed into 4x4 homogeneous
            T = np.eye(4)
            T[:3, :3] = R

            transforms.append(T)

    return transforms


def split_by_determinant(transforms):
    proper = []
    improper = []

    for T in transforms:
        det = np.linalg.det(T[:3, :3])
        if np.isclose(det, 1.0):
            proper.append(T)
        else:
            improper.append(T)

    return proper, improper


parser = argparse.ArgumentParser(description="3D GLB Viewer")

parser.add_argument(
    "--dir",
    type=str,
    default="../bags/new_model_dino/",
    help="Path to bags dir",
)

parser.add_argument(
    "--file",
    type=str,
    default="outdoor.glb",
    help="Path to the .glb file to view",
)

parser.add_argument(
    "--env",
    type=str,
    default="outdoor",
)

parser.add_argument("--robot", type=str, default="bunker")

parser.add_argument("--aug", type=str, default="no_aug")

parser.add_argument("--algo", type=str, default="bridger")

parser.add_argument(
    "--ref",
    type=str,
    default="reference_bunker_outdoor_reference_trial_1.bag",
    help="Name of the reference .bag file to read. Bag files are expected to be in ../bags/",
)

parser.add_argument(
    "--trial",
    type=str,
    default="1",
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
dir = args.dir


scene = trimesh.Scene()

mesh = trimesh.load(glb_file)

if isinstance(mesh, trimesh.Scene):
    scene = mesh  # Keep as scene
else:
    scene = trimesh.Scene(mesh)


# Uncomment to add axes at mesh origin
# mesh_origin = mesh.centroid  # or use mesh.bounds[0] for corner
# axis_mesh = trimesh.creation.axis(
#     origin_size=0.1,  # Size of origin sphere
#     axis_radius=0.02,  # Thickness of axes
#     axis_length=2.0,  # Length of each axis
#     transform=trimesh.transformations.translation_matrix(mesh_origin),
# )
# scene.add_geometry(axis_mesh, node_name="mesh_axes")


# all_transforms = generate_all_axis_aligned_transforms()
# proper_rotations, reflections = split_by_determinant(all_transforms)

# print(all_transforms[24])

# R_world_to_opencv = all_transforms[24]
# # 24

R_world_to_opencv = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
)

# TODO ADD CROSS
offset_reference_trajectory = [13.7, -0.4, -8.0]
if "dino" in algo:
    offset = [13.7, -0.4, -8.0]
    angle = 80
elif "nohist" in algo:
    offset = [13.7, -0.2, -8.0]
    angle = 80
elif "vint" in algo:
    offset = [13.7, -0.2, -8.0]
    angle = 80
elif "nomad" in algo:
    offset = [13.7, -0.4, -8.0]
    angle = 80
elif "metnet" in algo:
    offset = [13.7, -0.2, -7.8]
    angle = 80
else:
    angle = 30
    offset = [
        0.0,
        1.0,
        4.0,
    ]  # third value is too offset in z to align start position with robot position in mesh


if "dino" in algo:
    color_name = "lightcoral"
    algo_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)
elif "nohist" in algo:
    color_name = "cyan"
    algo_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)
elif "vint" in algo:
    color_name = "sienna"  # brown
    algo_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)
elif "nomad" in algo:
    color_name = "purple"
    algo_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)
elif "metnet" in algo:
    color_name = "deeppink"
    algo_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)


# Processing ALL trajectories

trials = args.trial.split(" ")  # Allow multiple trials separated by space

for trial in trials:
    bag = rosbag.Bag(f"../bags/{dir}{algo}_{robot}_{env}_{aug}_trial_{trial}.bag")
    # reset
    poses = []
    positions_corrected = []
    path = None

    for _, msg, _ in bag.read_messages(topics=["/laser_odometry"]):
        p = msg.pose.pose.position
        poses.append([p.x, p.y, p.z])

    bag.close()

    poses = np.array(poses)
    positions_corrected = poses
    positions_corrected = rotate_trajectory(positions_corrected, angle, axis="z")

    positions_corrected = positions_corrected @ R_world_to_opencv[:3, :3].T

    # NOW apply offset in OpenCV coordinate system
    positions_corrected[:, :] += offset

    path = trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
        vertices=positions_corrected,
        colors=[algo_color],
    )
    scene.add_geometry(path, node_name=f"trajectory_trial_{trial}")


# Processing reference trajectory

bag_ref = rosbag.Bag(f"{dir}{ref_bag_file}")

poses_ref = []

# Processing reference trajectory
for _, msg, _ in bag_ref.read_messages(topics=["/laser_odometry"]):
    p = msg.pose.pose.position
    poses_ref.append([p.x, p.y, p.z])

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


angle = 80
positions_ref_corrected = poses_ref
positions_ref_corrected = rotate_trajectory(positions_ref_corrected, angle, axis="z")

# Transform to OpenCV coordinates FIRST
positions_ref_corrected = positions_ref_corrected @ R_world_to_opencv[:3, :3].T

# NOW apply offset in OpenCV coordinate system
positions_ref_corrected[:, :] += offset_reference_trajectory

color_name = "chartreuse"
# 2. Convert to RGBA (0.0 - 1.0) then to (0 - 255)
ref_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)


# reference
path_ref = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_ref_corrected)))],
    vertices=positions_ref_corrected,
    colors=[ref_color],  # green for reference
)


# Add reference trajectory to scene
scene.add_geometry(path_ref, node_name="reference_trajectory")

scene.show()
