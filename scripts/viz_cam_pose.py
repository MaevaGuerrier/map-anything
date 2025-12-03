import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd

# --- minimal example robot pose ---

TRAJ_DIR = "../images"
REF_TRAJ = "gnm_bunker_mist_office_sharp_no_aug_trial_1"
CSV_FILE = "nodes.csv"


csv_path = f"{TRAJ_DIR}/{REF_TRAJ}/{CSV_FILE}"
df = pd.read_csv(csv_path)


# We viz first robot pose
x, y, z = df.loc[0, ["node_x_odom", "node_y_odom", "node_z_odom"]]
qx, qy, qz, qw = df.loc[0, ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]]

robot_pos = np.array([x, y, z])
robot_quat = np.array([qx, qy, qz, qw])

# --- camera offset in robot frame (meters) ---
offset_camera = np.array([0.19, 0.0, 0.135]) # cam horizontal offset forward 19cm, horizontal offset 13.5cm

T_robot_to_world = np.eye(4)
# T_robot_to_world[:3, 3] = robot_pos
# T_robot_to_world[:3, :3] = R.from_quat(robot_quat).as_matrix()
print("T_robot_to_world", T_robot_to_world)


# rotate offset to world
T_robot_to_camera = np.array([
    [1, 0, 0, 0.19],
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.135],
    [0, 0, 0, 1]
])
T_camera_to_world = T_robot_to_world @ T_robot_to_camera
print(T_camera_to_world)

# camera_pos = robot_pos # + offset_world # offset camera position from robot center
# camera_quat = robot_quat  # same orientation works as long as camera is not tilted


# world x forward
# y is right 
# z is down


R_world_to_opencv = np.array([
    [ 0,  1,  0,  0],
    [ 0,  0, -1,  0],
    [ 1,  0,  0,  0],
    [ 0,  0,  0,  1]
])


camera_poses = T_camera_to_world @ R_world_to_opencv
print("camera_poses", camera_poses)
# r_world_to_cam = np.array([
#                             [ 0,  1,  0],   # x is right 
#                             [ 0, 0,  1],   # y is down 
#                             [ 1,  0,  0],   # z forward
#                             [ 0,  0,  0] # last row translation
#                         ])

# Twr_camera_frame = Twr_camera @ r_world_to_cam 
# print("Twr_camera_frame", Twr_camera_frame)




# --- helper: plot a coordinate frame ---
def plot_frame(ax, origin, Rot_mat, scale=0.1):
    colors = ['r', 'g', 'b']
    Rot_mat_T = Rot_mat[:3, :3].T
    for i in range(3):
        ax.quiver(
            origin[0], origin[1], origin[2],
            Rot_mat_T[0, i], Rot_mat_T[1, i], Rot_mat_T[2, i],
            color=colors[i], linewidth=2
        )

# --- plotting ---
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# plot world frame
# plot_frame(ax, np.array([0,0,0]), [0,0,0,1], scale=0.15)

# plot robot frame
plot_frame(ax, T_robot_to_world[:3, 3], T_robot_to_world[:3, :3], scale=0.15)

# plot camera frame
plot_frame(ax, camera_poses[:3, 3], camera_poses[:3, :3], scale=0.15)

# annotate
ax.text(*T_robot_to_world[:3, 3], "Robot", color="k")
ax.text(*camera_poses[:3, 3], "Camera", color="k")
# set limits
s = 0.5
ax.set_xlim(-s, s)
ax.set_ylim(-s, s)
ax.set_zlim(0, s)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("World (RGB), Robot (RGB), Camera (RGB) Frames")

plt.show()
