import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from mapanything.utils.image import preprocess_inputs
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
import argparse
import rerun as rr

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

# https://github.com/openxrlab/xrprimer/blob/main/docs/en/transform/camera_convention.md
# odom orientation (quaternion x, y, z ,w) 

def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
        )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )



def compute_camera_pose(robot_pos, robot_quat):
    """
    Returns camera pose as a 4x4 cam2world matrix.
    """
    # world_T_robot
    Twr = np.eye(4) # T_robot_to_world = np.eye(4)
    Twr[:3, :3] = R.from_quat(robot_quat).as_matrix()
    Twr[:3, 3] = robot_pos

    T_robot_to_camera = np.array([
        [1, 0, 0, 0.19],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.135],
        [0, 0, 0, 1]
    ])

    Twr = Twr @ T_robot_to_camera


    R_world_to_opencv = np.array([
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ])

    camera_poses = Twr @ R_world_to_opencv

    return camera_poses

def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )

    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.glb",
        help="Output path for GLB file (default: output.glb)",
    )

    return parser



def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()


    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
    # For Apache 2.0 license model, use "facebook/map-anything-apache"
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    TRAJ_DIR = "../images"
    REF_TRAJ = "gnm_bunker_mist_office_sharp_no_aug_trial_1"
    CSV_FILE = "nodes.csv"

    # height = 480
    # width = 640
    # channels = 3

    csv_path = f"{TRAJ_DIR}/{REF_TRAJ}/{CSV_FILE}"
    df = pd.read_csv(csv_path)

    last_idx = df["node_idx"].max() 

    views_example = []

    for idx in range(0, last_idx + 1):
        if idx not in [0,1,2,3]:
            continue  

        x, y, z = df.loc[idx, ["node_x_odom", "node_y_odom", "node_z_odom"]]
        qx, qy, qz, qw = df.loc[idx, ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]]
        
        robot_pos = np.array([x, y, z])
        robot_quat = np.array([qx, qy, qz, qw])
        
        img_idx = df.loc[idx, "node_idx"]
        img_path = f"{TRAJ_DIR}/{REF_TRAJ}/{img_idx}.png"
        try:
            image = cv2.imread(img_path)  # shape (H, W, 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            break
        
        print(f"Processing camera poses and images: {img_path}")
        # OpenCV convention (XYZ - RDF)X is right, Y is down, Z is forward
        camera_poses = compute_camera_pose(robot_pos, robot_quat)  # 4x4 matrix

        # we need to https://github.com/facebookresearch/map-anything/issues/9
        views_example.append(
            {
                "img": image,              # (H, W, 3)
                "camera_poses": camera_poses,  # (4, 4) or tuple of (quats, trans) in OpenCV cam2world convention
                "is_metric_scale": torch.tensor([True], device=device)
            },
        )

    processed_views = preprocess_inputs(views_example)

    print("Running inference...")
    start_time = time.time()
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=False,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )
    print(f"Inference complete {time.time() - start_time:.2f} seconds.")

    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(predictions):
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        # Compute new pts3d using depth, intrinsics, and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")

        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)

        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)

        # Save GLB file
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()