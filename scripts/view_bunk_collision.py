import trimesh
import trimesh.transformations as tf
from trimesh.path.entities import Line


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


if __name__ == "__main__":
    obj, bbox_path = fig_bunker_collision()

    scale_factor = 0.6  # shrink the object to 60%

    scale = tf.scale_matrix(scale_factor)

    obj.apply_transform(scale)
    bbox_path.apply_transform(scale)

    scene = trimesh.Scene(obj)
    scene.add_geometry(bbox_path, node_name="bunker_collision_bbox")

    scene.show()
