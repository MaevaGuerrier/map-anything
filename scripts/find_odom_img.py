import pandas as pd
import numpy as np
import os
import rosbag
from typing import Any, Tuple, List, Dict
from PIL import Image
import rosbag # conda install -c conda-forge ros-rosbag
from cv_bridge import CvBridge
import cv2

import os
import glob



# TODO 

# DF TIME IMAGE NUMBER
# DF TIME ODOM 
# MERGE ASOF TIME NEAREST DROP IMAGE w/o odom



def get_process_func(img_func_name: str, odom_func_name: str):
    img_process_func = globals()[img_func_name]
    odom_process_func = globals()[odom_func_name]
    return img_process_func, odom_process_func

def process_images(im_list: List, img_process_func, output_path=None) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg, output_dir=output_path)
        images.append(img)
    return images


# Slow but works
def get_latest_png_index(directory: str) -> int or None:
    """
    Returns the largest numeric prefix of *.png files in the directory.
    Example: for 0.png, 1.png -> returns 1.

    If no numeric PNG exists, returns None.
    """
    png_files = glob.glob(os.path.join(directory, "*.png"))

    indices = []
    for f in png_files:
        base = os.path.basename(f)
        name, _ = os.path.splitext(base)

        if name.isdigit():
            indices.append(int(name))

    if not indices:
        return None

    return max(indices)



def process_custom_img(msg, output_dir=None) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    if output_dir is not None:
        idx = get_latest_png_index(output_dir)
        counter = 0 if idx is None else idx + 1
        pil_image.save(os.path.join(output_dir, f"{counter}.png"))
        print(f"Saved image {counter}.png to {output_dir}")
    return pil_image

def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    zs = []
    orientations = {"x": [], "y": [], "z": [], "w": []}
    for odom_msg in odom_list:
        xy, z, ox, oy, oz, ow = odom_process_func(odom_msg, ang_offset) # return [position.x, position.y], yaw, orientation.x, orientation.y, orientation.z, orientation.w
        xys.append(xy)
        zs.append(z)
        orientations["x"].append(ox)
        orientations["y"].append(oy)
        orientations["z"].append(oz)
        orientations["w"].append(ow)
    return {"position": np.array(xys), "z": np.array(zs), "orientation": {k: np.array(v) for k, v in orientations.items()}}

def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw

def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float, float, float, float, float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], position.z, orientation.x, orientation.y, orientation.z, orientation.w



def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = .66,
    ang_offset: float = 0.0,
    output_path: str = None,
) -> Tuple[List, Dict[np.ndarray, np.ndarray]]:
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
            print(f"Odom msg data: pos=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}), orient=({msg.pose.pose.orientation.x}, {msg.pose.pose.orientation.y}, {msg.pose.pose.orientation.z}, {msg.pose.pose.orientation.w})")
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func, output_path=output_path)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    return img_data, traj_data


def get_bag_img_traj_data(bag, img_topic, odom_topic, process_images_fn, process_odom_fn, output_path=None):

    bag_img_data, bag_traj_data = get_images_and_odom(
        bag=bag,
        imtopics=img_topic,
        odomtopics=odom_topic,
        img_process_func=process_images_fn,
        odom_process_func=process_odom_fn,
        output_path=output_path,
    )

    # bag_img_data is a list of images
    # bag_traj_data is a  dictionary -> dict_keys(['position', 'yaw']
    assert len(bag_img_data) == len(bag_traj_data['position']), "Number of images and odometry entries must match"
    return bag_img_data, bag_traj_data


def associate_topomap_node_with_odom(bag_img_data, bag_traj_data):
    """
    Associates each image node index with odometry (x, y) and saves results.
    Returns a pandas DataFrame with columns:
        [image_index, pose_x, pose_y]
    """
    positions = bag_traj_data["position"]
    zs = bag_traj_data["z"]
    orientations_x = bag_traj_data["orientation"]["x"]
    orientations_y = bag_traj_data["orientation"]["y"]
    orientations_z = bag_traj_data["orientation"]["z"]
    orientations_w = bag_traj_data["orientation"]["w"]

    n_imgs = len(bag_img_data)
    print(f"Number of images: {n_imgs}")

    df = pd.DataFrame({
        "node_idx": range(n_imgs),
        "node_x_odom": [pos[0] for pos in positions[:n_imgs]],
        "node_y_odom": [pos[1] for pos in positions[:n_imgs]],
        "node_z_odom": zs[:n_imgs],
        "orientation_x": orientations_x[:n_imgs],
        "orientation_y": orientations_y[:n_imgs],
        "orientation_z": orientations_z[:n_imgs],
        "orientation_w": orientations_w[:n_imgs],
    })
    return df

if __name__ == "__main__":

    IMAGE_TOPIC = "/oak/rgb/image_raw"
    ODOM_TOPIC = "/lvi_sam/lidar/mapping/odometry"

    bag_dir = "../bags"
    bag_name = "gnm_bunker_mist_office_sharp_no_aug_trial_1.bag"

    bag_path = os.path.join(bag_dir, bag_name)
    bag = rosbag.Bag(bag_path, "r")

    #TODO Delete if exists or ignore
    output_path = f"../images/{bag_name.replace('.bag', '')}"
    os.makedirs(output_path, exist_ok=True)

    bag_img_data, bag_traj_data = get_bag_img_traj_data(
        bag=bag,
        img_topic=IMAGE_TOPIC,
        odom_topic=ODOM_TOPIC,
        process_images_fn=process_custom_img,
        process_odom_fn=nav_to_xy_yaw,
        output_path=output_path,
    )

    node_df = associate_topomap_node_with_odom(
        bag_img_data=bag_img_data,
        bag_traj_data=bag_traj_data
    )


    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "nodes.csv")
    node_df.to_csv(save_path, index=False)