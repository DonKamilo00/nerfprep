"""
nerfprep.datasets.colmap
========================
Parse COLMAP sparse reconstruction output into clean Python objects.

Supports both binary (.bin) and text (.txt) formats, auto-detected.

Usage:
    from nerfprep.datasets.colmap import load_colmap

    scene = load_colmap("path/to/sparse/0")
    print(scene.cameras)      # dict of Camera objects
    print(scene.images)       # dict of Image objects
    print(scene.points3D)     # dict of Point3D objects
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    A COLMAP camera model with intrinsic parameters.

    Attributes:
        id:         Unique camera ID
        model:      COLMAP camera model name (e.g. 'PINHOLE', 'OPENCV')
        width:      Image width in pixels
        height:     Image height in pixels
        params:     Model-dependent intrinsic parameters (see COLMAP docs)
                    PINHOLE:  [fx, fy, cx, cy]
                    SIMPLE_PINHOLE: [f, cx, cy]
                    OPENCV:   [fx, fy, cx, cy, k1, k2, p1, p2]
    """
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        """Focal length x. Works for PINHOLE and OPENCV models."""
        return float(self.params[0])

    @property
    def fy(self) -> float:
        """Focal length y. For SIMPLE_PINHOLE, same as fx."""
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        """Principal point x."""
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        """Principal point y."""
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[2])
        return float(self.params[3])

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx,    0.0,  self.cx],
            [   0.0,  self.fy, self.cy],
            [   0.0,     0.0,     1.0],
        ], dtype=np.float64)


@dataclass
class Image:
    """
    A COLMAP image with extrinsic pose (world-to-camera convention).

    COLMAP stores poses as quaternion (qw, qx, qy, qz) + translation t,
    where the transform is:  x_cam = R @ x_world + t

    Attributes:
        id:           Unique image ID
        qvec:         Quaternion [qw, qx, qy, qz] (world-to-camera)
        tvec:         Translation [tx, ty, tz] (world-to-camera)
        camera_id:    ID of the associated Camera
        name:         Image filename
        xys:          Nx2 array of 2D keypoint locations
        point3D_ids:  N array of corresponding 3D point IDs (-1 if unmatched)
    """
    id: int
    qvec: np.ndarray          # shape (4,)  [qw, qx, qy, qz]
    tvec: np.ndarray          # shape (3,)
    camera_id: int
    name: str
    xys: np.ndarray           # shape (N, 2)
    point3D_ids: np.ndarray   # shape (N,)

    @property
    def R(self) -> np.ndarray:
        """3x3 rotation matrix (world-to-camera)."""
        return qvec_to_rotmat(self.qvec)

    @property
    def t(self) -> np.ndarray:
        """Translation vector (world-to-camera), shape (3,)."""
        return self.tvec

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates, shape (3,)."""
        return -self.R.T @ self.tvec

    @property
    def c2w(self) -> np.ndarray:
        """4x4 camera-to-world transform matrix."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R.T
        mat[:3,  3] = self.camera_center
        return mat

    @property
    def w2c(self) -> np.ndarray:
        """4x4 world-to-camera transform matrix."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R
        mat[:3,  3] = self.tvec
        return mat


@dataclass
class Point3D:
    """
    A COLMAP 3D point with color and reprojection error.

    Attributes:
        id:         Unique point ID
        xyz:        3D position [x, y, z]
        rgb:        Color [r, g, b] in range [0, 255]
        error:      Mean reprojection error in pixels
        image_ids:  IDs of images that observe this point
        point2D_idxs: Index into each image's keypoint list
    """
    id: int
    xyz: np.ndarray          # shape (3,)
    rgb: np.ndarray          # shape (3,)  uint8
    error: float
    image_ids: np.ndarray    # shape (N,)
    point2D_idxs: np.ndarray # shape (N,)


@dataclass
class ColmapScene:
    """
    A complete COLMAP sparse reconstruction.

    Attributes:
        cameras:  Dict mapping camera_id -> Camera
        images:   Dict mapping image_id  -> Image
        points3D: Dict mapping point_id  -> Point3D
    """
    cameras:  Dict[int, Camera]
    images:   Dict[int, Image]
    points3D: Dict[int, Point3D]

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points(self) -> int:
        return len(self.points3D)

    def point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the full point cloud as arrays.

        Returns:
            xyz: (N, 3) float64 array of 3D positions
            rgb: (N, 3) uint8  array of colors
        """
        pts = list(self.points3D.values())
        xyz = np.stack([p.xyz for p in pts], axis=0)
        rgb = np.stack([p.rgb for p in pts], axis=0)
        return xyz, rgb

    def summary(self) -> str:
        lines = [
            "ColmapScene",
            f"  cameras  : {self.num_cameras}",
            f"  images   : {self.num_images}",
            f"  points3D : {self.num_points}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# COLMAP camera model parameter counts
# ---------------------------------------------------------------------------

CAMERA_MODEL_PARAMS: Dict[str, int] = {
    "SIMPLE_PINHOLE": 3,   # f, cx, cy
    "PINHOLE": 4,           # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,    # f, cx, cy, k
    "RADIAL": 5,            # f, cx, cy, k1, k2
    "OPENCV": 8,            # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE": 8,   # fx, fy, cx, cy, k1, k2, k3, k4
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

# Map COLMAP model ID (int) -> model name string
CAMERA_MODEL_IDS: Dict[int, str] = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.
    Matches COLMAP's convention exactly.
    """
    qw, qx, qy, qz = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].
    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


# ---------------------------------------------------------------------------
# Binary parsers
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id      = struct.unpack("<i", f.read(4))[0]
            model_id    = struct.unpack("<i", f.read(4))[0]
            width       = struct.unpack("<Q", f.read(8))[0]
            height      = struct.unpack("<Q", f.read(8))[0]
            model_name  = CAMERA_MODEL_IDS[model_id]
            num_params  = CAMERA_MODEL_PARAMS[model_name]
            params      = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name,
                width=int(width), height=int(height), params=params,
            )
    return cameras


def _read_images_bin(path: Path) -> Dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id    = struct.unpack("<i", f.read(4))[0]
            qvec      = np.array(struct.unpack("<4d", f.read(32)))  # qw qx qy qz
            tvec      = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<i", f.read(4))[0]
            # Read null-terminated filename
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")
            # Read 2D keypoints
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys_and_ids  = struct.unpack(f"<{3*num_points2D}d", f.read(24 * num_points2D))
            xys         = np.array(xys_and_ids).reshape(-1, 3)[:, :2]
            point3D_ids = np.array(xys_and_ids).reshape(-1, 3)[:, 2].astype(np.int64)
            images[img_id] = Image(
                id=img_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids,
            )
    return images


def _read_points3D_bin(path: Path) -> Dict[int, Point3D]:
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pt_id  = struct.unpack("<Q", f.read(8))[0]
            xyz    = np.array(struct.unpack("<3d", f.read(24)))
            rgb    = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error  = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track_data = struct.unpack(f"<{2*track_len}i", f.read(8 * track_len))
            track = np.array(track_data).reshape(-1, 2)
            points[pt_id] = Point3D(
                id=int(pt_id), xyz=xyz, rgb=rgb, error=error,
                image_ids=track[:, 0], point2D_idxs=track[:, 1],
            )
    return points


# ---------------------------------------------------------------------------
# Text parsers
# ---------------------------------------------------------------------------

def _read_cameras_txt(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id     = int(parts[0])
            model_name = parts[1]
            width      = int(parts[2])
            height     = int(parts[3])
            params     = np.array([float(p) for p in parts[4:]])
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name,
                width=width, height=height, params=params,
            )
    return cameras


def _read_images_txt(path: Path) -> Dict[int, Image]:
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # Images are stored in pairs of lines
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id    = int(parts[0])
        qvec      = np.array([float(x) for x in parts[1:5]])
        tvec      = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name      = parts[9]
        # Second line: x y point3D_id triplets
        pt_parts  = lines[i + 1].split()
        if pt_parts:
            pts = np.array([float(x) for x in pt_parts]).reshape(-1, 3)
            xys         = pts[:, :2]
            point3D_ids = pts[:, 2].astype(np.int64)
        else:
            xys         = np.zeros((0, 2))
            point3D_ids = np.zeros(0, dtype=np.int64)
        images[img_id] = Image(
            id=img_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=name,
            xys=xys, point3D_ids=point3D_ids,
        )
    return images


def _read_points3D_txt(path: Path) -> Dict[int, Point3D]:
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            pt_id  = int(parts[0])
            xyz    = np.array([float(x) for x in parts[1:4]])
            rgb    = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            error  = float(parts[7])
            track  = np.array([int(x) for x in parts[8:]], dtype=np.int64).reshape(-1, 2)
            points[pt_id] = Point3D(
                id=pt_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=track[:, 0], point2D_idxs=track[:, 1],
            )
    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_colmap(
    sparse_dir: str | Path,
    load_points: bool = True,
) -> ColmapScene:
    """
    Load a COLMAP sparse reconstruction from a directory.

    Automatically detects binary (.bin) or text (.txt) format.
    The directory should contain cameras, images, and points3D files.

    Args:
        sparse_dir:   Path to the COLMAP sparse directory (e.g. "sparse/0")
        load_points:  Whether to load the 3D point cloud. Set False to skip
                      if you only need camera poses (faster for large scenes).

    Returns:
        ColmapScene with .cameras, .images, .points3D

    Example:
        scene = load_colmap("data/scene/sparse/0")
        print(scene.summary())
        xyz, rgb = scene.point_cloud()
    """
    sparse_dir = Path(sparse_dir)
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP directory not found: {sparse_dir}")

    # Auto-detect format
    if (sparse_dir / "cameras.bin").exists():
        fmt = "bin"
    elif (sparse_dir / "cameras.txt").exists():
        fmt = "txt"
    else:
        raise FileNotFoundError(
            f"No cameras.bin or cameras.txt found in {sparse_dir}. "
            "Make sure this is a valid COLMAP sparse directory."
        )

    if fmt == "bin":
        cameras  = _read_cameras_bin(sparse_dir / "cameras.bin")
        images   = _read_images_bin(sparse_dir / "images.bin")
        points3D = _read_points3D_bin(sparse_dir / "points3D.bin") if load_points else {}
    else:
        cameras  = _read_cameras_txt(sparse_dir / "cameras.txt")
        images   = _read_images_txt(sparse_dir / "images.txt")
        points3D = _read_points3D_txt(sparse_dir / "points3D.txt") if load_points else {}

    return ColmapScene(cameras=cameras, images=images, points3D=points3D)
