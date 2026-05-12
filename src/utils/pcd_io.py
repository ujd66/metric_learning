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

    Supports two naming conventions:
    - Legacy: {base}_heightmap.png, {base}_info.txt, {base}_transform.json
    - Label:  {base}.png, {base}_meta.json
    """
    base = os.path.splitext(os.path.basename(path))[0]
    directory = os.path.dirname(path)

    # For *_crop.pcd files, the meta.json uses the stem without "_crop"
    # e.g. 000029_0001_crop.pcd -> 000029_0001_meta.json
    stem = base.removesuffix("_crop")

    info = {
        "pcd_path": path,
        "heightmap_path": None,
        "info_path": None,
        "transform_path": None,
        "info_text": None,
        "transform": None,
    }

    # heightmap / image: legacy, same-base png, stem png
    for candidate in (f"{base}_heightmap.png", f"{base}.png", f"{stem}.png"):
        p = os.path.join(directory, candidate)
        if os.path.exists(p):
            info["heightmap_path"] = p
            break

    # info text
    info_file = os.path.join(directory, f"{base}_info.txt")
    if os.path.exists(info_file):
        info["info_path"] = info_file
        with open(info_file) as f:
            info["info_text"] = f.read().strip()

    # transform / meta json: legacy, base meta, stem meta
    for candidate in (f"{base}_transform.json", f"{base}_meta.json", f"{stem}_meta.json"):
        p = os.path.join(directory, candidate)
        if os.path.exists(p):
            info["transform_path"] = p
            with open(p) as f:
                info["transform"] = json.load(f)
            break

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
