"""
nerfprep.camera
===============
Camera math utilities for neural rendering pipelines.

Covers:
  - PinholeCamera  : project, unproject, ray generation, frustum checks
  - convert_convention : transform poses between OpenCV/OpenGL/COLMAP/Blender
  - Rays           : dataclass holding ray origins + directions

Coordinate convention reference
--------------------------------
OPENCV  (COLMAP default) : X right, Y down,    Z forward
OPENGL  (NeRF / instant-ngp) : X right, Y up,  Z backward
BLENDER                  : X right, Y forward,  Z up

Usage:
    from nerfprep.camera import PinholeCamera, convert_convention, Rays
    from nerfprep.datasets.colmap import load_colmap

    scene  = load_colmap("sparse/0")
    colmap_cam = scene.cameras[1]
    colmap_img = scene.images[1]

    cam  = PinholeCamera.from_colmap(colmap_cam)
    c2w  = convert_convention(colmap_img.c2w, src="opencv", dst="opengl")
    rays = cam.generate_rays(c2w)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Convention = Literal["opencv", "opengl", "blender", "colmap"]

# ---------------------------------------------------------------------------
# Convention converter
# ---------------------------------------------------------------------------

# Each convention is defined by how its axes map to a shared world frame.
# We store a 3x3 matrix that rotates FROM that convention TO a canonical frame
# (we use the same canonical as OpenCV / COLMAP for simplicity).
#
# The matrices below flip axes to convert between conventions.

_TO_OPENCV: dict[str, np.ndarray] = {
    # OpenCV is the canonical frame — identity
    "opencv": np.eye(3),
    "colmap": np.eye(3),   # COLMAP uses OpenCV convention natively
    # OpenGL: Y up, Z backward  →  flip Y and Z
    "opengl":  np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float64),
    # Blender: Y forward, Z up  →  permute axes
    "blender": np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=np.float64),
}


def convert_convention(
    c2w: np.ndarray,
    src: Convention,
    dst: Convention,
) -> np.ndarray:
    """
    Convert a 4x4 camera-to-world matrix between coordinate conventions.

    Args:
        c2w:  (4, 4) camera-to-world matrix in `src` convention
        src:  Source convention. One of: 'opencv', 'opengl', 'blender', 'colmap'
        dst:  Destination convention.

    Returns:
        (4, 4) camera-to-world matrix in `dst` convention

    Examples:
        # COLMAP poses → NeRF / instant-ngp (OpenGL)
        c2w_nerf = convert_convention(c2w_colmap, src="opencv", dst="opengl")

        # Blender camera export → COLMAP
        c2w_colmap = convert_convention(c2w_blender, src="blender", dst="opencv")
    """
    if src not in _TO_OPENCV:
        raise ValueError(f"Unknown convention '{src}'. Choose from {list(_TO_OPENCV)}")
    if dst not in _TO_OPENCV:
        raise ValueError(f"Unknown convention '{dst}'. Choose from {list(_TO_OPENCV)}")

    if c2w.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix, got {c2w.shape}")

    if src == dst:
        return c2w.copy()

    # Convert: src → opencv → dst
    # For the rotation block:  R_dst = R_dst_from_opencv @ R_opencv_from_src @ R_src
    R_to_cv   = _TO_OPENCV[src]           # src  → opencv
    R_from_cv = _TO_OPENCV[dst].T         # opencv → dst  (transpose = inverse for orthogonal)

    conv = np.eye(4, dtype=np.float64)
    conv[:3, :3] = R_from_cv @ R_to_cv

    return conv @ c2w


# ---------------------------------------------------------------------------
# Rays dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rays:
    """
    A bundle of rays, each defined by an origin and unit direction.

    Attributes:
        origins:    (H, W, 3) or (N, 3) ray origins in world space
        directions: (H, W, 3) or (N, 3) unit ray directions in world space
    """
    origins:    np.ndarray   # (..., 3)
    directions: np.ndarray   # (..., 3)

    def __post_init__(self):
        if self.origins.shape != self.directions.shape:
            raise ValueError(
                f"origins and directions must have the same shape. "
                f"Got {self.origins.shape} vs {self.directions.shape}"
            )

    def __len__(self) -> int:
        return self.origins.shape[0]

    @property
    def shape(self) -> tuple:
        return self.origins.shape

    def reshape(self, *shape) -> "Rays":
        """Reshape the ray bundle e.g. from (H, W, 3) to (H*W, 3)."""
        return Rays(
            origins=self.origins.reshape(*shape, 3),
            directions=self.directions.reshape(*shape, 3),
        )

    def at(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate ray positions at parameter t.

        Args:
            t: scalar or (...) array of distances along ray

        Returns:
            (..., 3) world-space positions: origin + t * direction
        """
        return self.origins + t[..., None] * self.directions


# ---------------------------------------------------------------------------
# PinholeCamera
# ---------------------------------------------------------------------------

class PinholeCamera:
    """
    A pinhole camera model with projection, unprojection and ray generation.

    All operations assume OpenCV convention internally (X right, Y down, Z fwd).
    Use convert_convention() on poses before passing them to methods that
    take a c2w matrix if your poses are in a different convention.

    Args:
        fx, fy:       Focal lengths in pixels
        cx, cy:       Principal point in pixels
        width, height: Image dimensions in pixels
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ):
        self.fx     = float(fx)
        self.fy     = float(fy)
        self.cx     = float(cx)
        self.cy     = float(cy)
        self.width  = int(width)
        self.height = int(height)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_colmap(cls, camera) -> "PinholeCamera":
        """
        Construct from a nerfprep Camera object (from load_colmap).

        Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV.
        Note: distortion parameters are ignored — this is a pinhole model.

        Example:
            scene = load_colmap("sparse/0")
            cam   = PinholeCamera.from_colmap(scene.cameras[1])
        """
        return cls(
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            width=camera.width,
            height=camera.height,
        )

    @classmethod
    def from_fov(
        cls,
        fov_x: float,
        width: int,
        height: int,
        fov_in_degrees: bool = True,
    ) -> "PinholeCamera":
        """
        Construct from horizontal field of view.

        Args:
            fov_x:          Horizontal field of view
            width, height:  Image dimensions in pixels
            fov_in_degrees: If True, fov_x is in degrees, else radians
        """
        if fov_in_degrees:
            fov_x = np.deg2rad(fov_x)
        fx = (width / 2.0) / np.tan(fov_x / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx,    0.0,  self.cx],
            [   0.0,  self.fy, self.cy],
            [   0.0,     0.0,     1.0],
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse of the 3x3 intrinsic matrix."""
        return np.linalg.inv(self.K)

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * np.arctan2(self.width / 2.0, self.fx)

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * np.arctan2(self.height / 2.0, self.fy)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def __repr__(self) -> str:
        return (
            f"PinholeCamera(fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
            f"width={self.width}, height={self.height})"
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points (in camera space) to 2D pixel coordinates.

        Args:
            points_3d: (..., 3) array of 3D points in camera coordinates.
                       Points behind the camera (Z <= 0) are projected but
                       will land outside the image — use in_frustum() to filter.

        Returns:
            (..., 2) array of pixel coordinates [u, v]
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        Z = points_3d[..., 2:3]
        # Perspective divide
        xy_norm = points_3d[..., :2] / np.where(np.abs(Z) > 1e-8, Z, 1e-8)
        u = self.fx * xy_norm[..., 0:1] + self.cx
        v = self.fy * xy_norm[..., 1:2] + self.cy
        return np.concatenate([u, v], axis=-1)

    def project_world(
        self,
        points_world: np.ndarray,
        w2c: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D world-space points to 2D pixel coordinates.

        Args:
            points_world: (..., 3) points in world coordinates
            w2c:          (4, 4) world-to-camera matrix (OpenCV convention)

        Returns:
            (..., 2) pixel coordinates [u, v]
        """
        points_world = np.asarray(points_world, dtype=np.float64)
        shape = points_world.shape
        pts_h = np.concatenate([
            points_world.reshape(-1, 3),
            np.ones((points_world.reshape(-1, 3).shape[0], 1))
        ], axis=-1)              # (N, 4)
        pts_cam = (w2c @ pts_h.T).T[:, :3]   # (N, 3)
        return self.project(pts_cam).reshape(*shape[:-1], 2)

    def unproject(
        self,
        depth: np.ndarray,
        pixels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Unproject a depth map (or specific pixels) to 3D camera-space points.

        Args:
            depth:  (H, W) depth map in the same units as your scene,
                    OR (N,) depths if pixels is provided.
            pixels: (N, 2) pixel coordinates [u, v] to unproject.
                    If None, unprojection is done for every pixel in the image.

        Returns:
            (H, W, 3) or (N, 3) 3D points in camera space
        """
        depth = np.asarray(depth, dtype=np.float64)

        if pixels is not None:
            pixels = np.asarray(pixels, dtype=np.float64)
            x = (pixels[..., 0] - self.cx) / self.fx
            y = (pixels[..., 1] - self.cy) / self.fy
            return np.stack([x * depth, y * depth, depth], axis=-1)
        else:
            H, W = depth.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
            x = (u.astype(np.float64) - self.cx) / self.fx
            y = (v.astype(np.float64) - self.cy) / self.fy
            return np.stack([x * depth, y * depth, depth], axis=-1)

    # ------------------------------------------------------------------
    # Frustum checks
    # ------------------------------------------------------------------

    def in_frustum(
        self,
        points_cam: np.ndarray,
        near: float = 0.01,
        far: float = 1e6,
    ) -> np.ndarray:
        """
        Return boolean mask of which camera-space points are inside the frustum.

        Args:
            points_cam: (..., 3) points in camera space
            near, far:  Depth clipping planes

        Returns:
            (...) boolean array, True where point is visible
        """
        uv = self.project(points_cam)
        Z  = points_cam[..., 2]
        in_depth  = (Z > near) & (Z < far)
        in_width  = (uv[..., 0] >= 0) & (uv[..., 0] < self.width)
        in_height = (uv[..., 1] >= 0) & (uv[..., 1] < self.height)
        return in_depth & in_width & in_height

    # ------------------------------------------------------------------
    # Ray generation
    # ------------------------------------------------------------------

    def generate_rays(
        self,
        c2w: np.ndarray,
        pixels: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> Rays:
        """
        Generate world-space rays from this camera given a pose.

        This is the core operation for NeRF / volume rendering pipelines.
        The c2w matrix should be in the same convention as the camera model
        (OpenCV by default). Use convert_convention() if needed.

        Args:
            c2w:       (4, 4) camera-to-world matrix
            pixels:    (N, 2) specific pixel [u, v] coordinates to generate
                       rays for. If None, generates rays for all pixels.
            normalize: If True, ray directions are unit vectors (recommended).

        Returns:
            Rays with .origins and .directions of shape (H, W, 3) or (N, 3)

        Example:
            # All rays in image
            rays = cam.generate_rays(c2w)        # (H, W, 3)
            flat = rays.reshape(-1)               # (H*W, 3)

            # Rays for specific pixels (e.g. random sample for NeRF batch)
            coords = np.random.randint([0,0], [H,W], size=(1024, 2))
            rays   = cam.generate_rays(c2w, pixels=coords)  # (1024, 3)
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must be (4,4), got {c2w.shape}")

        R   = c2w[:3, :3]
        cam_origin = c2w[:3, 3]  # camera center in world space

        if pixels is not None:
            pixels = np.asarray(pixels, dtype=np.float64)
            # pixels: (N, 2) — [u, v]
            x = (pixels[:, 0] - self.cx) / self.fx
            y = (pixels[:, 1] - self.cy) / self.fy
            dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (N, 3)
            dirs_world = (R @ dirs_cam.T).T                          # (N, 3)
            origins = np.broadcast_to(cam_origin, dirs_world.shape).copy()
        else:
            # Generate for every pixel
            u = np.arange(self.width,  dtype=np.float64)
            v = np.arange(self.height, dtype=np.float64)
            uu, vv = np.meshgrid(u, v, indexing="xy")    # both (H, W)
            x = (uu - self.cx) / self.fx
            y = (vv - self.cy) / self.fy
            ones = np.ones_like(x)
            dirs_cam   = np.stack([x, y, ones], axis=-1)            # (H, W, 3)
            dirs_world = (R @ dirs_cam.reshape(-1, 3).T).T          # (H*W, 3)
            dirs_world = dirs_world.reshape(self.height, self.width, 3)
            origins    = np.broadcast_to(cam_origin, dirs_world.shape).copy()

        if normalize:
            norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True)
            dirs_world = dirs_world / np.where(norms > 1e-8, norms, 1e-8)

        return Rays(origins=origins, directions=dirs_world)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pixel_grid(self) -> np.ndarray:
        """
        Return (H, W, 2) array of all pixel [u, v] coordinates.
        Useful for vectorized operations over the whole image.
        """
        u = np.arange(self.width,  dtype=np.float64)
        v = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        return np.stack([uu, vv], axis=-1)

    def rescale(self, scale: float) -> "PinholeCamera":
        """
        Return a new camera scaled to a different resolution.
        Useful for multi-scale training.

        Args:
            scale: Scale factor, e.g. 0.5 for half resolution

        Returns:
            New PinholeCamera at scaled resolution
        """
        return PinholeCamera(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=int(self.width  * scale),
            height=int(self.height * scale),
        )
