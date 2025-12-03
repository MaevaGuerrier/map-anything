import trimesh
import argparse

parser = argparse.ArgumentParser(description="3D GLB Viewer")
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="Path to the .glb file to view",
)
args = parser.parse_args()

# Path to your .glb file
scene = trimesh.load(args.file)

# If it's a scene, combine all meshes into one
if isinstance(scene, trimesh.Scene):
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    combined_mesh = trimesh.util.concatenate(meshes)
else:
    combined_mesh = scene

# Display
combined_mesh.show()