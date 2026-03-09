import argparse
import itertools

import numpy as np
import rosbag  # pip install bagpy https://stackoverflow.com/questions/59794328/importing-rosbag-in-python-3
import trimesh
from trimesh.path.entities import Line
import trimesh.transformations as tf


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


def fig_bunker_collision():
    bunker = "bunker.glb"

    # Path to your .glb file
    obj = trimesh.load(bunker)

    bbox = obj.bounding_box_oriented
    corners = bbox.vertices  # 8 corner points

    import numpy as np

    # Edges of a cube (12 edges)
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical
        ]
    )

    entities = [Line(e) for e in edges]

    red = [255, 0, 0, 255]

    bbox_path = trimesh.path.Path3D(
        vertices=corners, entities=entities, colors=[red] * len(entities)
    )

    return obj, bbox_path


parser = argparse.ArgumentParser(description="3D GLB Viewer")
parser.add_argument(
    "--file",
    type=str,
    default="easy_office.glb",
    help="Path to the .glb file to view",
)

parser.add_argument(
    "--env",
    type=str,
    default="easy_office",
)

parser.add_argument("--robot", type=str, default="bunker")

parser.add_argument("--aug", type=str, default="no_aug")

parser.add_argument("--algo", type=str, default="bridger")

parser.add_argument(
    "--ref",
    type=str,
    default="reference_bunker_easy_office_reference_trial_1.bag",
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
offset_reference_trajectory = [-1.5, 0.4, 2.3]
angle_ref = -10
if "bridger" in algo:
    offset = [-1.5, 0.4, 2.3]  # navibridger
    angle = -10
elif "gnm" in algo:
    offset = [-1.5, 0.4, 2.3]  # gnm
    angle = -10
elif "vint" in algo:
    offset = [-1.5, 0.4, 2.3]  # vint
    angle = -10
elif "nomad" in algo:
    offset = [-1.5, 0.4, 2.3]  # nomad
    angle = -10
elif "cross" in algo:
    offset = [-1.5, 0.4, 2.3]  # cross
    angle = -10
else:
    angle = 30
    offset = [
        0.0,
        1.0,
        4.0,
    ]  # third value is too offset in z to align start position with robot position in mesh


if "bridger" in algo:
    algo_color = [255, 0, 0, 255]  # red
elif "gnm" in algo:
    algo_color = [0, 0, 255, 255]  # blue
elif "vint" in algo:
    algo_color = [255, 165, 0, 255]  # orange
elif "nomad" in algo:
    algo_color = [255, 150, 255, 255]  # pink
elif "cross" in algo:
    algo_color = [0, 255, 255, 255]  # cyan


# Processing ALL trajectories

trials = args.trial.split(" ")  # Allow multiple trials separated by space

last_poses = {"trial_1": None, "trial_2": None, "trial_3": None}

for trial in trials:
    bag = rosbag.Bag(f"../bags/{algo}_{robot}_{env}_{aug}_trial_{trial}.bag")
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
    last_poses[f"trial_{trial}"] = positions_corrected[-1]

    path = trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
        vertices=positions_corrected,
        colors=[algo_color],
    )
    scene.add_geometry(path, node_name=f"trajectory_trial_{trial}")


# Processing reference trajectory

bag_ref = rosbag.Bag(f"../bags/{ref_bag_file}")

poses_ref = []

# Processing reference trajectory
for _, msg, _ in bag_ref.read_messages(topics=["/laser_odometry"]):
    p = msg.pose.pose.position
    poses_ref.append([p.x, p.y, p.z])

bag_ref.close()

poses_ref = np.array(poses_ref)


positions_ref_corrected = poses_ref
positions_ref_corrected = rotate_trajectory(
    positions_ref_corrected, angle_ref, axis="z"
)

# Transform to OpenCV coordinates FIRST
positions_ref_corrected = positions_ref_corrected @ R_world_to_opencv[:3, :3].T

# NOW apply offset in OpenCV coordinate system
positions_ref_corrected[:, :] += offset_reference_trajectory


# reference
path_ref = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_ref_corrected)))],
    vertices=positions_ref_corrected,
    colors=[[0, 255, 0, 255]],  # green for reference
)


# Add reference trajectory to scene
scene.add_geometry(path_ref, node_name="reference_trajectory")


if collision:
    scale_factor = 0.73
    if "cross" in algo:
        angle = np.deg2rad(120)
        target_positions = [positions_corrected[-1]]
    elif "nomad" in algo:
        target_positions = [last_poses["trial_1"], last_poses["trial_2"]]
        angle = np.deg2rad(100)
    elif "bridger" in algo:
        target_positions = [positions_corrected[-1]]
        angle = np.deg2rad(120)
        target_positions = [last_poses["trial_1"]]
    else:
        angle = np.deg2rad(100)
        target_positions = [positions_corrected[-1]]

    for i, target_pos in enumerate(target_positions):
        # if i == 1:
        #     angle = np.deg2rad(60)  for nomad trial 2, which is less rotated than trial 1, so we apply smaller angle to better align bunker with trajectory

        # IMPORTANT: reload object each time (avoid cumulative transforms)
        obj, bbox_path = fig_bunker_collision()

        target_pos = target_pos.copy()

        S = tf.scale_matrix(scale_factor)
        R = tf.rotation_matrix(angle, [0, 1, 0])
        T = tf.translation_matrix(target_pos)

        M = T @ R @ S

        obj.apply_transform(M)
        bbox_path.apply_transform(M)

        # Adjust to ground level
        z_offset = 0.2
        shift_y = 0.02

        obj.apply_translation([-shift_y, -z_offset, 0])
        bbox_path.apply_translation([-shift_y, -z_offset, 0])

        scene.add_geometry(obj, node_name=f"bunker_collision_{i}")
        # scene.add_geometry(bbox_path, node_name=f"bunker_collision_bbox_{i}")


scene.show()
