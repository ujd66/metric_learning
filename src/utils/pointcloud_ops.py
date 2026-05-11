import numpy as np


def read_pcd(path):
    """Read a .pcd file using Open3D, return points [N, 3+]."""
    from src.utils.pcd_io import read_pcd as _read_pcd
    return _read_pcd(path)


def load_pointcloud(path):
    """Load point cloud from .pcd or .npy file, return numpy array [N, C]."""
    if path.endswith(".pcd"):
        return read_pcd(path)

    raw = np.load(path, allow_pickle=True)
    try:
        data = raw.item()
        if isinstance(data, dict) and "points" in data:
            return np.array(data["points"], dtype=np.float64)
    except (ValueError, AttributeError):
        pass
    return np.array(raw, dtype=np.float64)


def sample_points(points, num_points):
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
    else:
        indices = np.concatenate([
            np.arange(N),
            np.random.choice(N, num_points - N, replace=True),
        ])
    return points[indices]


def normalize_points(points):
    centroid = points[:, :3].mean(axis=0)
    points[:, :3] -= centroid
    max_dist = np.linalg.norm(points[:, :3], axis=1).max()
    if max_dist > 0:
        points[:, :3] /= max_dist
    return points


def random_jitter(points, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*points[:, :3].shape), -clip, clip)
    points[:, :3] += noise
    return points


def random_z_rotation(points):
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0],
                         [sin_a, cos_a, 0],
                         [0, 0, 1]], dtype=np.float64)
    points[:, :3] = points[:, :3] @ rotation.T
    return points


def random_dropout(points, ratio=0.1):
    N = points.shape[0]
    keep = int(N * (1 - ratio))
    indices = np.random.choice(N, keep, replace=False)
    return points[indices]
