# pip install trimesh pyglet
import argparse
import trimesh


parser = argparse.ArgumentParser(description="3D GLB Viewer")
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="Path to the .glb file to view",
)
args = parser.parse_args()

# Path to your .glb file
mesh = trimesh.load(args.file)

# Open an interactive 3D viewer window
mesh.show()
