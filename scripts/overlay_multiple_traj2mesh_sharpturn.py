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


def make_mesh_transform(translate, rotate_deg):
    T = np.eye(4)
    # Apply rotations in x→y→z order
    for angle_deg, axis in zip(rotate_deg, ["x", "y", "z"]):
        if angle_deg == 0.0:
            continue
        a = np.deg2rad(angle_deg)
        c, s = np.cos(a), np.sin(a)
        R4 = np.eye(4)
        if axis == "x":
            R4[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif axis == "y":
            R4[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == "z":
            R4[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T = R4 @ T
    T[:3, 3] = translate
    return T


def filter_stationary_points(positions, min_dist=0.01):
    dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    keep = np.concatenate([[True], dists > min_dist])
    return positions[keep]


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
    default="sharpturn.glb",
    help="Path to the .glb file to view",
)

parser.add_argument(
    "--env",
    type=str,
    default="sharpturn",
)

parser.add_argument("--robot", type=str, default="bunker")

parser.add_argument("--aug", type=str, default="no_aug")

parser.add_argument("--algo", type=str, default="bridger")

parser.add_argument(
    "--ref",
    type=str,
    default="reference_bunker_sharpturn_reference_trial_1.bag",
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

parser.add_argument(
    "--mesh-translate",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("TX", "TY", "TZ"),
    help="Translate mesh by (tx, ty, tz)",
)

parser.add_argument(
    "--mesh-rotate",
    type=float,
    nargs=3,
    default=[0.0, -50.0, 0.0],
    metavar=("RX", "RY", "RZ"),
    help="Rotate mesh by (rx, ry, rz) degrees around x, y, z axes",
)

parser.add_argument(
    "--traj-rotate-x",
    type=float,
    default=150.0,
    help="Rotate trajectory around X axis (degrees)",
)

parser.add_argument(
    "--traj-rotate-y",
    type=float,
    default=-1.0,  # up down
    help="Rotate trajectory around Y axis (degrees)",
)

parser.add_argument(
    "--traj-rotate-z",
    type=float,
    default=-20.0,
    help="Rotate trajectory around Z axis (degrees)",
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

# if isinstance(mesh, trimesh.Scene):
#     scene = mesh  # Keep as scene
# else:
#     scene = trimesh.Scene(mesh)

mesh_transform = make_mesh_transform(args.mesh_translate, args.mesh_rotate)

if isinstance(mesh, trimesh.Scene):
    scene = mesh
    # Apply transform to each geometry in the scene
    for name, geom in scene.geometry.items():
        geom.apply_transform(mesh_transform)
else:
    mesh.apply_transform(mesh_transform)
    scene = trimesh.Scene(mesh)

rotations = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
    np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
    np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
]

R_world_to_opencv = rotations[1]

# ================= OFFSET SETUP ====================
offset_reference_trajectory = [-1.6, -1.12, 0.0]  # forward, side , side toward me

if aug == "4hz04":

    if "dino" in algo:
        offset =  [-1.9, -1.1, 0.0]
        rotate_x = 150.0
        rotate_y = -2.12
        rotate_z = -20.0
    elif "vint" in algo:
        offset = [-2., -1., 0.0]
        rotate_x = 150.0
        rotate_y = -5.12
        rotate_z = -20.0
    elif "metnet" in algo:
        offset =  [-2, -1.2, 0.0]
        rotate_x = 150.0
        rotate_y = -5.
        rotate_z = -10.0
if aug == "2hz04":

    if "dino" in algo:
        offset =  [-2.2, -1.1, 0.0]
        rotate_x = 150.0
        rotate_y = -2.12
        rotate_z = -20.0
    elif "vint" in algo:
        offset = [-2.3, -1., 0.0]
        rotate_x = 150.0
        rotate_y = -5.12
        rotate_z = -20.0

if aug == "1hz04":

    if "dino" in algo:
        offset =  [-2.2, -1.1, 0.0]
        rotate_x = 150.0
        rotate_y = -2.12
        rotate_z = -20.0
    elif "vint" in algo:
        offset = [-2.5, -1.12, 0.0]
        rotate_x = 150.0
        rotate_y = -5.12
        rotate_z = -20.0

if aug == "4hz02":

    if "dino" in algo:
        offset =  [-2.45, -1.1, 0.0]
        rotate_x = 150.0
        rotate_y = -2.12
        rotate_z = -20.0
    elif "vint" in algo:
        offset = [-2.5, -1.12, 0.0]
        rotate_x = 150.0
        rotate_y = -5.12
        rotate_z = -20.0

if aug == "4hz01":

    if "dino" in algo:
        offset =  [-2.2, -1.1, 0.0]
        rotate_x = 150.0
        rotate_y = -2.12
        rotate_z = -20.0
    elif "vint" in algo:
        offset = [-2.5, -1.12, 0.0]
        rotate_x = 150.0
        rotate_y = -5.12
        rotate_z = -20.0

# =====================================================


# =========== COLOR SETUP ====================

color_name = "chartreuse"
# 2. Convert to RGBA (0.0 - 1.0) then to (0 - 255)
ref_color = (np.array(mcolors.to_rgba(color_name)) * 255).astype(np.uint8)


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

# ============================================


# ==== ACTUAL TRAJS ====

trials = args.trial.split(" ")  # Allow multiple trials separated by space

for trial in trials:
    bag = rosbag.Bag(f"{dir}{algo}_{robot}_{env}_{aug}_trial_{trial}.bag")
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
    positions_corrected = positions_corrected @ R_world_to_opencv[:3, :3].T

    # NOW apply offset in OpenCV coordinate system
    positions_corrected[:, :] += offset

    positions_corrected = rotate_trajectory(
        positions_corrected, rotate_x, axis="x"
    )
    positions_corrected = rotate_trajectory(
        positions_corrected, rotate_y, axis="y"
    )
    positions_corrected = rotate_trajectory(
        positions_corrected, rotate_z, axis="z"
    )

    path = trimesh.path.Path3D(
        entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
        vertices=positions_corrected,
        colors=[algo_color],
    )
    scene.add_geometry(path, node_name=f"trajectory_trial_{trial}")






# ==== REF =====

bag_ref = rosbag.Bag(f"{dir}{ref_bag_file}")

poses_ref = []

# Processing reference trajectory
for _, msg, _ in bag_ref.read_messages(topics=["/laser_odometry"]):
    p = msg.pose.pose.position
    poses_ref.append([p.x, p.y, p.z])

bag_ref.close()

poses_ref = np.array(poses_ref)


positions_ref_corrected = poses_ref

# Transform to OpenCV coordinates FIRST
positions_ref_corrected = positions_ref_corrected @ R_world_to_opencv[:3, :3].T

# NOW apply offset in OpenCV coordinate system
positions_ref_corrected[:, :] += offset_reference_trajectory

positions_ref_corrected = rotate_trajectory(
    positions_ref_corrected, args.traj_rotate_x, axis="x"
)
positions_ref_corrected = rotate_trajectory(
    positions_ref_corrected, args.traj_rotate_y, axis="y"
)
positions_ref_corrected = rotate_trajectory(
    positions_ref_corrected, args.traj_rotate_z, axis="z"
)


# reference
path_ref = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_ref_corrected)))],
    vertices=positions_ref_corrected,
    colors=[ref_color],  # green for reference
)


# Add reference trajectory to scene
scene.add_geometry(path_ref, node_name="reference_trajectory")
# axis = trimesh.creation.axis(origin_size=0.1)
# scene.add_geometry(axis)


# my_point = positions_ref_corrected[0]

# # 2. Create a transformation matrix for that point
# # This creates a 4x4 matrix that moves things to [x, y, z]
# translation = trimesh.transformations.translation_matrix(my_point)

# 3. Create the axis at that specific location
# point_axis = trimesh.creation.axis(
#     origin_size=0.05,
#     axis_length=0.5,
#     transform=translation,  # This "sticks" the axis to your point
# )

# # 4. Add to your existing scene
# scene.add_geometry(point_axis, node_name="point_marker")


scene.show()
