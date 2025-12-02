import trimesh
import open3d as o3d
import numpy as np
import rosbag

# ============================================================
# 5. LOAD ROSBAG ODOMETRY (DEBUGGED VERSION)
# ============================================================
bag = rosbag.Bag("../bags/gnm_bunker_mist_office_sharp_no_aug_trial_1.bag")

poses = []
timestamps = []

print("Reading bag file...")
message_count = 0

for topic, msg, t in bag.read_messages(topics=["/lvi_sam/lidar/mapping/odometry"]):
    p = msg.pose.pose.position
    poses.append([p.x, p.y, p.z])
    timestamps.append(t.to_sec())
    message_count += 1

bag.close()

print(f"Total odometry messages read: {message_count}")

if message_count == 0:
    print("ERROR: No messages found! Check the topic name.")
    # Let's find what topics exist:
    bag = rosbag.Bag("../bags/reference_bunker_mist_office_sharp_reference_trial_1.bag")
    topics = bag.get_type_and_topic_info()[1].keys()
    print("Available topics:")
    for topic in topics:
        print(f"  {topic}")
    bag.close()
    exit()

poses = np.array(poses)
timestamps = np.array(timestamps)

# Check for duplicate positions (robot not moving)
unique_poses = np.unique(poses, axis=0)
print(f"Unique positions: {len(unique_poses)} out of {len(poses)}")

# Check time span
print(f"Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Average message rate: {len(poses) / (timestamps[-1] - timestamps[0]):.2f} Hz")

# Visualize the path in 2D to check shape
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot(poses[:, 0], poses[:, 1], 'b.-', linewidth=0.5, markersize=1)
plt.plot(poses[0, 0], poses[0, 1], 'go', markersize=10, label='Start')
plt.plot(poses[-1, 0], poses[-1, 1], 'ro', markersize=10, label='End')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title(f'Odometry Path (XY plane) - {len(poses)} points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print(f"\nPath statistics:")
print(f"  Start position: {poses[0]}")
print(f"  End position: {poses[-1]}")
print(f"  Total distance traveled: {np.sum(np.linalg.norm(np.diff(poses, axis=0), axis=1)):.2f} m")
