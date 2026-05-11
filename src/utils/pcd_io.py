import json
import os

import numpy as np


def read_pcd(path):
    """Read a .pcd file using Open3D and return points as numpy array [N, 3+].

    Returns xyz columns. If Open3D can also read intensity, returns [N, 4]
    with columns [x, y, z, intensity].
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float64)

    if points.size == 0:
        raise RuntimeError(f"Open3D read 0 points from {path}")

    return points


def read_pcd_xyz(path):
    """Read a .pcd file and return only xyz coordinates as [N, 3]."""
    return read_pcd(path)


def read_pcd_with_intensity(path):
    """Read a .pcd file and return xyz + intensity as [N, 4].

    Falls back to [N, 3] (xyz only) if intensity is not available.
    """
    import open3d as o3d

    try:
        cloud = o3d.t.io.read_point_cloud(path)
        positions = cloud.point.positions.numpy().astype(np.float64)

        if "intensity" in cloud.point:
            intensity = cloud.point.intensity.numpy().astype(np.float64)
            if intensity.ndim == 1:
                intensity = intensity.reshape(-1, 1)
            return np.hstack([positions, intensity])

        return positions
    except Exception:
        return read_pcd(path)


def get_pcd_info(path):
    """Read companion files for a PCD sample.

    Given 000003.pcd, looks for 000003_info.txt, 000003_transform.json,
    and 000003_heightmap.png in the same directory.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    directory = os.path.dirname(path)

    info = {
        "pcd_path": path,
        "heightmap_path": None,
        "info_path": None,
        "transform_path": None,
        "info_text": None,
        "transform": None,
    }

    heightmap = os.path.join(directory, f"{base}_heightmap.png")
    if os.path.exists(heightmap):
        info["heightmap_path"] = heightmap

    info_file = os.path.join(directory, f"{base}_info.txt")
    if os.path.exists(info_file):
        info["info_path"] = info_file
        with open(info_file) as f:
            info["info_text"] = f.read().strip()

    transform_file = os.path.join(directory, f"{base}_transform.json")
    if os.path.exists(transform_file):
        info["transform_path"] = transform_file
        with open(transform_file) as f:
            info["transform"] = json.load(f)

    return info


def read_pcd_header(path):
    """Parse the ASCII header of a .pcd file, return dict of header fields."""
    header = {}
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("DATA"):
                header["DATA"] = line.split()[1]
                break
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            key = parts[0]
            values = parts[1:]
            if len(values) == 1:
                header[key] = values[0]
            else:
                header[key] = values
    return header
