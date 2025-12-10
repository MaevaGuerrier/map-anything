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
from scipy.spatial.transform import Rotation as R

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



def get_orientation_opencv(orientation : list):
    # ROS typically uses: X-forward, Y-left, Z-up
    # OpenCV typically uses: X-right, Y-down, Z-forward
    R_world_to_opencv = np.array([
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])

    R_wc = R_world_to_opencv[:3, :3]   # world → camera rotation

    R_obj_world = R.from_quat(orientation).as_matrix()

    # world → camera transform
    R_obj_cam = R_wc @ R_obj_world

    # back to quaternion
    return R.from_matrix(R_obj_cam).as_quat()




parser = argparse.ArgumentParser(description="3D GLB Viewer")
parser.add_argument(
    "--file",
    type=str,
    default="office.glb",
    help="Path to the .glb file to view",
)

parser.add_argument(
    "--bag",
    type=str,
    required=True,
    help="Name of the .bag file to read. Bag files are expected to be in ../bags/",
)

parser.add_argument(
    "--ref",
    type=str,
    default="reference_bunker_office_loop_reference_trial_1.bag",
    help="Name of the reference .bag file to read. Bag files are expected to be in ../bags/",
)

parser.add_argument(
    "--col",
    action="store_true",
    help="Mark collision points in the trajectory",
)


args = parser.parse_args()




glb_file = args.file
bag_file = args.bag
ref_bag_file = args.ref
collision = args.col


if not glb_file.endswith(".glb"):
    raise ValueError("The provided file is not a .glb file.")

if not bag_file.endswith(".bag"):
    bag_file += ".bag"

if not ref_bag_file.endswith(".bag"):
    ref_bag_file += ".bag"


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




bag = rosbag.Bag(f"../bags/{bag_file}")
bag_ref = rosbag.Bag(f"../bags/{ref_bag_file}")
poses = []
# orientations = []
poses_ref = []


# we only store orientation to orient the bunker collision object later if needed
for _, msg, _ in bag.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
    p = msg.pose.pose.position
    # o = msg.pose.pose.orientation
    poses.append([p.x, p.y, p.z])
    # orientations.append([o.x, o.y, o.z, o.w])

bag.close()

poses = np.array(poses)

# Processing reference trajectory
for _, msg, _ in bag_ref.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
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



R_world_to_opencv = np.array([
    [ 0,  1,  0,  0],
    [ 0,  0, -1,  0],
    [ 1,  0,  0,  0],
    [ 0,  0,  0,  1]
])


positions_corrected = poses @ R_world_to_opencv[:3, :3].T
positions_ref_corrected = poses_ref @ R_world_to_opencv[:3, :3].T



if "bridger" in bag_file:
    algo_color = [255, 0, 0, 255] # red
elif "gnm" in bag_file:
    algo_color = [0, 0, 255, 255] # blue
elif "vint" in bag_file:
    algo_color = [255, 165, 0, 255] # orange
elif "nomad" in bag_file:
    algo_color = [255, 192, 203, 255] # pink


# TODO make it config file
# if glb_file == "office.glb":
offset_reference_trajectory = [2.5, -0.5, -1.45]

if "bridger" in bag_file:
    offset = [2.5, -0.5, -1.45] #navibridger
elif "gnm" in bag_file:
    offset = [2.5, -0.5, -1.45] # gnm
elif "vint" in bag_file:
    offset = [2.5, -0.5, -1.45] # vint
elif "nomad" in bag_file:
    offset = [2.70, -0.5, -1.40] # nomad
else:
    offset = [2.5, -0.5, -1.45] # third value is too offset in z to align start position with robot position in mesh




# print(positions_corrected.shape) # (1588, 3)
positions_corrected[:, :] += offset
positions_ref_corrected[:, :] += offset_reference_trajectory


path = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_corrected)))],
    vertices=positions_corrected,
    colors=[algo_color]
)


if collision:
    target_pos = positions_corrected[-1]
    # target_orientation = get_orientation_opencv(orientations[-1])
    obj, bbox_path = fig_bunker_collision()
    # reducing size
    # scale_factor = 0.70   # shrink the object to 70%
    scale_obj = tf.scale_matrix(0.70)
    scale_bbox = tf.scale_matrix(0.60)

    obj.apply_transform(scale_obj)
    bbox_path.apply_transform(scale_bbox)

    # Needs to be after rotation
    obj.apply_translation(target_pos)
    bbox_path.apply_translation(target_pos)

    scene.add_geometry(obj, node_name="bunker_collision")
    scene.add_geometry(bbox_path, node_name="bunker_collision_bbox")



scene.add_geometry(path, node_name="trajectory")


# reference
path_ref = trimesh.path.Path3D(
    entities=[trimesh.path.entities.Line(np.arange(len(positions_ref_corrected)))],
    vertices=positions_ref_corrected,
    colors=[[0, 255, 0, 255]]  # green for reference
)

# Add reference trajectory to scene
scene.add_geometry(path_ref, node_name="reference_trajectory")


# Uncomment to add axes
# trajectory_end = positions_corrected[-1]
# axis_traj_end = trimesh.creation.axis(
#     origin_size=0.15,
#     axis_radius=0.04,
#     axis_length=3.0,
#     transform=trimesh.transformations.translation_matrix(trajectory_end)
# )
# scene.add_geometry(axis_traj_end, node_name="trajectory_end_axes")



transform_rotation = np.array([
    [1, 0, 0, 0],  # x red
    [0, 1, 0, 0],  # y green
    [0, 0, 1, 0],  # z blue
    [0, 0, 0, 1]
])

scene.apply_transform(transform_rotation)


scene.show()
