"""
nerfprep.io
===========
Format conversion utilities for neural rendering pipelines.

Functions
---------
- save_transforms_json : COLMAP scene → instant-ngp / vanilla NeRF format
- load_transforms_json : Load transforms.json → list of frames
- save_cameras_json    : COLMAP scene → nerfstudio cameras.json
- load_cameras_json    : Load nerfstudio cameras.json
- save_ply             : Save point cloud as PLY file
- load_ply             : Load PLY point cloud → xyz, rgb arrays

Format references
-----------------
transforms.json : https://github.com/NVlabs/instant-ngp
cameras.json    : https://docs.nerf.studio/reference/api/cameras.html
PLY             : Standard binary/ASCII polygon file format

Usage
-----
    from nerfprep.datasets.colmap import load_colmap
    from nerfprep.io import save_transforms_json, save_ply

    scene = load_colmap("sparse/0")

    # For instant-ngp or vanilla NeRF training
    save_transforms_json(scene, "transforms.json")

    # Save point cloud for inspection in MeshLab / CloudCompare
    xyz, rgb = scene.point_cloud()
    save_ply(xyz, rgb, "pointcloud.ply")
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion [qw, qx, qy, qz]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


def _c2w_opencv_to_opengl(c2w: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 c2w from OpenCV convention (COLMAP) to OpenGL convention.
    instant-ngp and most NeRF codebases expect OpenGL convention.
    OpenCV → OpenGL: flip Y and Z axes.
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return flip @ c2w


# ---------------------------------------------------------------------------
# transforms.json  (instant-ngp / vanilla NeRF format)
# ---------------------------------------------------------------------------

def save_transforms_json(
    scene,
    output_path: str | Path,
    image_dir: str = "images",
    split: Optional[str] = None,
    aabb_scale: int = 16,
) -> None:
    """
    Convert a COLMAP scene to transforms.json (instant-ngp / NeRF format).

    The output follows the instant-ngp convention:
      - Poses are 4x4 camera-to-world matrices in OpenGL convention
        (X right, Y up, Z backward)
      - Camera intrinsics stored at top level (assumes single camera)
      - Each frame has a file_path and transform_matrix

    Args:
        scene:       ColmapScene from load_colmap()
        output_path: Path to write transforms.json
        image_dir:   Relative path prefix for image filenames in the JSON
        split:       If provided, appended to image stem for train/val/test
                     splits (e.g. split="train" → "images/frame_001_train.jpg")
        aabb_scale:  Scene scale for instant-ngp (2, 4, 8, 16, or 32).
                     Set larger for unbounded outdoor scenes.

    Example:
        scene = load_colmap("sparse/0")
        save_transforms_json(scene, "transforms.json")
        # Or for train split:
        save_transforms_json(scene, "transforms_train.json", split="train")
    """
    output_path = Path(output_path)

    # Use the first camera for intrinsics (assumes shared intrinsics)
    cam = next(iter(scene.cameras.values()))

    data: Dict = {
        "aabb_scale":   aabb_scale,
        "fl_x":         cam.fx,
        "fl_y":         cam.fy,
        "cx":           cam.cx,
        "cy":           cam.cy,
        "w":            cam.width,
        "h":            cam.height,
        "camera_model": cam.model,
        "frames":       [],
    }

    for img in sorted(scene.images.values(), key=lambda x: x.name):
        # c2w in OpenCV convention from COLMAP
        c2w_opencv = img.c2w
        # Convert to OpenGL convention for NeRF frameworks
        c2w_opengl = _c2w_opencv_to_opengl(c2w_opencv)

        file_path = f"{image_dir}/{img.name}"

        frame = {
            "file_path":        file_path,
            "transform_matrix": c2w_opengl.tolist(),
        }
        data["frames"].append(frame)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_transforms_json(
    path: str | Path,
) -> Dict:
    """
    Load a transforms.json file into a Python dict.

    Returns the raw dict with top-level intrinsics and a 'frames' list.
    Each frame has 'file_path' and 'transform_matrix' (4x4 as nested list).

    Args:
        path: Path to transforms.json

    Returns:
        Dict with keys: fl_x, fl_y, cx, cy, w, h, frames, ...

    Example:
        data   = load_transforms_json("transforms.json")
        frames = data["frames"]
        c2w    = np.array(frames[0]["transform_matrix"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"transforms.json not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    # Convert transform_matrix lists back to numpy arrays
    for frame in data.get("frames", []):
        frame["transform_matrix"] = np.array(frame["transform_matrix"])
    return data


# ---------------------------------------------------------------------------
# cameras.json  (nerfstudio format)
# ---------------------------------------------------------------------------

def save_cameras_json(
    scene,
    output_path: str | Path,
    image_dir: str = "images",
) -> None:
    """
    Convert a COLMAP scene to nerfstudio's cameras.json format.

    The nerfstudio format stores per-frame intrinsics and extrinsics
    explicitly, supporting variable intrinsics across frames.
    Poses are stored in OpenCV convention (matching COLMAP).

    Args:
        scene:       ColmapScene from load_colmap()
        output_path: Path to write cameras.json
        image_dir:   Relative path prefix for image filenames

    Example:
        scene = load_colmap("sparse/0")
        save_cameras_json(scene, "cameras.json")
    """
    output_path = Path(output_path)

    frames = []
    for img in sorted(scene.images.values(), key=lambda x: x.name):
        cam = scene.cameras[img.camera_id]
        c2w = img.c2w   # OpenCV convention — nerfstudio expects this

        frame = {
            "file_path":        f"{image_dir}/{img.name}",
            "fl_x":             cam.fx,
            "fl_y":             cam.fy,
            "cx":               cam.cx,
            "cy":               cam.cy,
            "w":                cam.width,
            "h":                cam.height,
            "camera_model":     cam.model,
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    data = {
        "camera_model": next(iter(scene.cameras.values())).model,
        "frames":       frames,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_cameras_json(
    path: str | Path,
) -> Dict:
    """
    Load a nerfstudio cameras.json file.

    Args:
        path: Path to cameras.json

    Returns:
        Dict with 'camera_model' and 'frames' list. Each frame has
        per-image intrinsics and a numpy transform_matrix.

    Example:
        data = load_cameras_json("cameras.json")
        for frame in data["frames"]:
            c2w = frame["transform_matrix"]   # already numpy array
            fx  = frame["fl_x"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cameras.json not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    for frame in data.get("frames", []):
        frame["transform_matrix"] = np.array(frame["transform_matrix"])
    return data


# ---------------------------------------------------------------------------
# PLY point cloud
# ---------------------------------------------------------------------------

def save_ply(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    output_path: str | Path = "pointcloud.ply",
    binary: bool = True,
) -> None:
    """
    Save a point cloud as a PLY file.

    Supports both binary (default, smaller/faster) and ASCII formats.
    Compatible with MeshLab, CloudCompare, Open3D, and Blender.

    Args:
        xyz:         (N, 3) float array of 3D point positions
        rgb:         (N, 3) uint8 or float array of colors.
                     float values are expected in [0, 1] and auto-converted.
                     If None, saves positions only.
        output_path: Output file path (should end in .ply)
        binary:      If True, writes binary little-endian PLY (recommended).
                     If False, writes ASCII PLY (human-readable but larger).

    Example:
        # From a COLMAP scene
        scene = load_colmap("sparse/0")
        xyz, rgb = scene.point_cloud()
        save_ply(xyz, rgb, "scene.ply")

        # Position-only
        save_ply(xyz, output_path="positions.ply")
    """
    output_path = Path(output_path)
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

    N = len(xyz)
    has_color = rgb is not None

    if has_color:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            # float [0,1] → uint8
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        if rgb.shape != (N, 3):
            raise ValueError(f"rgb must be (N, 3), got {rgb.shape}")

    # Build PLY header
    props = "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        props += ("property uchar red\n"
                  "property uchar green\n"
                  "property uchar blue\n")

    fmt    = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        f"ply\n"
        f"format {fmt}\n"
        f"element vertex {N}\n"
        f"{props}"
        f"end_header\n"
    )

    if binary:
        with open(output_path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(N):
                f.write(struct.pack("<fff",
                    float(xyz[i, 0]),
                    float(xyz[i, 1]),
                    float(xyz[i, 2]),
                ))
                if has_color:
                    f.write(struct.pack("<BBB",
                        int(rgb[i, 0]),
                        int(rgb[i, 1]),
                        int(rgb[i, 2]),
                    ))
    else:
        with open(output_path, "w") as f:
            f.write(header)
            for i in range(N):
                line = f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}"
                if has_color:
                    line += f" {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
                f.write(line + "\n")


def load_ply(
    path: str | Path,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a PLY point cloud into numpy arrays.

    Supports binary little-endian and ASCII PLY files.
    Reads x, y, z and optionally red, green, blue properties.

    Args:
        path: Path to .ply file

    Returns:
        Tuple of:
          xyz: (N, 3) float64 array of positions
          rgb: (N, 3) uint8 array of colors, or None if no color in file

    Example:
        xyz, rgb = load_ply("pointcloud.ply")
        print(xyz.shape)   # (N, 3)
        print(rgb.shape)   # (N, 3) or None
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Extract metadata from header
        n_vertices = 0
        is_binary  = False
        properties = []

        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("format binary"):
                is_binary = True
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (dtype_str, name)

        prop_names = [p[1] for p in properties]
        has_color  = all(c in prop_names for c in ("red", "green", "blue"))

        # Build struct format for binary reading
        dtype_map = {
            "float": "f", "float32": "f",
            "double": "d", "float64": "d",
            "uchar": "B", "uint8": "B",
            "int": "i", "int32": "i",
            "short": "h", "int16": "h",
        }
        struct_fmt = "<" + "".join(
            dtype_map.get(p[0], "f") for p in properties
        )
        struct_size = struct.calcsize(struct_fmt)

        xyz = np.zeros((n_vertices, 3), dtype=np.float64)
        rgb = np.zeros((n_vertices, 3), dtype=np.uint8) if has_color else None

        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")
        if has_color:
            r_idx = prop_names.index("red")
            g_idx = prop_names.index("green")
            b_idx = prop_names.index("blue")

        if is_binary:
            for i in range(n_vertices):
                values = struct.unpack(struct_fmt, f.read(struct_size))
                xyz[i, 0] = values[x_idx]
                xyz[i, 1] = values[y_idx]
                xyz[i, 2] = values[z_idx]
                if has_color:
                    rgb[i, 0] = values[r_idx]
                    rgb[i, 1] = values[g_idx]
                    rgb[i, 2] = values[b_idx]
        else:
            for i in range(n_vertices):
                parts  = f.readline().decode("ascii").strip().split()
                values = [float(p) for p in parts]
                xyz[i, 0] = values[x_idx]
                xyz[i, 1] = values[y_idx]
                xyz[i, 2] = values[z_idx]
                if has_color and rgb is not None:
                    rgb[i, 0] = int(values[r_idx])
                    rgb[i, 1] = int(values[g_idx])
                    rgb[i, 2] = int(values[b_idx])

    return xyz, rgb
