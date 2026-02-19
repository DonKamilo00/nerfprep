"""
tests/test_colmap.py
====================
Tests for the COLMAP parser using synthetically generated data.
Run with: pytest tests/test_colmap.py -v
"""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nerfprep.datasets.colmap import (
    Camera,
    ColmapScene,
    Image,
    Point3D,
    load_colmap,
    qvec_to_rotmat,
    rotmat_to_qvec,
)


# ---------------------------------------------------------------------------
# Synthetic data writers (mirrors the binary format exactly)
# ---------------------------------------------------------------------------

def _write_cameras_bin(path: Path, cameras: list[dict]):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<i", cam["id"]))
            f.write(struct.pack("<i", cam["model_id"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            params = cam["params"]
            f.write(struct.pack(f"<{len(params)}d", *params))


def _write_images_bin(path: Path, images: list[dict]):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<i", img["id"]))
            f.write(struct.pack("<4d", *img["qvec"]))
            f.write(struct.pack("<3d", *img["tvec"]))
            f.write(struct.pack("<i", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            xys = img["xys"]
            ids = img["point3D_ids"]
            n   = len(xys)
            f.write(struct.pack("<Q", n))
            for (x, y), pid in zip(xys, ids):
                f.write(struct.pack("<3d", x, y, float(pid)))


def _write_points3D_bin(path: Path, points: list[dict]):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points:
            f.write(struct.pack("<Q", pt["id"]))
            f.write(struct.pack("<3d", *pt["xyz"]))
            f.write(struct.pack("<3B", *pt["rgb"]))
            f.write(struct.pack("<d", pt["error"]))
            track = pt["track"]
            f.write(struct.pack("<Q", len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack("<2i", img_id, pt2d_idx))


def _write_cameras_txt(path: Path, cameras: list[dict]):
    with open(path, "w") as f:
        f.write("# Camera list\n")
        for cam in cameras:
            params_str = " ".join(str(p) for p in cam["params"])
            model_name = {1: "PINHOLE", 0: "SIMPLE_PINHOLE"}[cam["model_id"]]
            f.write(f"{cam['id']} {model_name} {cam['width']} {cam['height']} {params_str}\n")


def _write_images_txt(path: Path, images: list[dict]):
    with open(path, "w") as f:
        f.write("# Image list\n")
        for img in images:
            q = " ".join(str(x) for x in img["qvec"])
            t = " ".join(str(x) for x in img["tvec"])
            f.write(f"{img['id']} {q} {t} {img['camera_id']} {img['name']}\n")
            pts = []
            for (x, y), pid in zip(img["xys"], img["point3D_ids"]):
                pts.append(f"{x} {y} {pid}")
            f.write(" ".join(pts) + "\n")


def _write_points3D_txt(path: Path, points: list[dict]):
    with open(path, "w") as f:
        f.write("# 3D point list\n")
        for pt in points:
            xyz = " ".join(str(x) for x in pt["xyz"])
            rgb = " ".join(str(x) for x in pt["rgb"])
            track_str = " ".join(f"{i} {j}" for i, j in pt["track"])
            f.write(f"{pt['id']} {xyz} {rgb} {pt['error']} {track_str}\n")


# ---------------------------------------------------------------------------
# Shared synthetic scene fixture
# ---------------------------------------------------------------------------

SYNTHETIC_CAMERAS = [
    {"id": 1, "model_id": 1, "width": 1920, "height": 1080,
     "params": [800.0, 800.0, 960.0, 540.0]},
]

SYNTHETIC_IMAGES = [
    {"id": 1, "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0],
     "camera_id": 1, "name": "frame_001.jpg",
     "xys": [[100.0, 200.0], [300.0, 400.0]],
     "point3D_ids": [1, 2]},
    {"id": 2, "qvec": [0.9999, 0.01, 0.0, 0.0], "tvec": [0.1, 0.0, 0.0],
     "camera_id": 1, "name": "frame_002.jpg",
     "xys": [[150.0, 250.0]],
     "point3D_ids": [1]},
]

SYNTHETIC_POINTS = [
    {"id": 1, "xyz": [1.0, 2.0, 5.0], "rgb": [255, 128, 0],
     "error": 0.5, "track": [(1, 0), (2, 0)]},
    {"id": 2, "xyz": [-1.0, 0.5, 4.0], "rgb": [0, 255, 128],
     "error": 0.3, "track": [(1, 1)]},
]


@pytest.fixture
def binary_scene_dir(tmp_path):
    _write_cameras_bin(tmp_path / "cameras.bin", SYNTHETIC_CAMERAS)
    _write_images_bin(tmp_path / "images.bin", SYNTHETIC_IMAGES)
    _write_points3D_bin(tmp_path / "points3D.bin", SYNTHETIC_POINTS)
    return tmp_path


@pytest.fixture
def text_scene_dir(tmp_path):
    _write_cameras_txt(tmp_path / "cameras.txt", SYNTHETIC_CAMERAS)
    _write_images_txt(tmp_path / "images.txt", SYNTHETIC_IMAGES)
    _write_points3D_txt(tmp_path / "points3D.txt", SYNTHETIC_POINTS)
    return tmp_path


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

class TestCamera:
    def test_binary_camera_count(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.num_cameras == 1

    def test_text_camera_count(self, text_scene_dir):
        scene = load_colmap(text_scene_dir)
        assert scene.num_cameras == 1

    def test_camera_intrinsics(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        cam = scene.cameras[1]
        assert cam.model == "PINHOLE"
        assert cam.width == 1920
        assert cam.height == 1080
        assert cam.fx == pytest.approx(800.0)
        assert cam.fy == pytest.approx(800.0)
        assert cam.cx == pytest.approx(960.0)
        assert cam.cy == pytest.approx(540.0)

    def test_camera_K_matrix(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        K = scene.cameras[1].K
        assert K.shape == (3, 3)
        assert K[0, 0] == pytest.approx(800.0)  # fx
        assert K[1, 1] == pytest.approx(800.0)  # fy
        assert K[0, 2] == pytest.approx(960.0)  # cx
        assert K[1, 2] == pytest.approx(540.0)  # cy
        assert K[2, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Image / pose tests
# ---------------------------------------------------------------------------

class TestImage:
    def test_image_count(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.num_images == 2

    def test_image_name(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.images[1].name == "frame_001.jpg"
        assert scene.images[2].name == "frame_002.jpg"

    def test_identity_pose(self, binary_scene_dir):
        """Image 1 has identity rotation, should give identity R matrix."""
        scene = load_colmap(binary_scene_dir)
        img = scene.images[1]
        R = img.R
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_camera_center_identity(self, binary_scene_dir):
        """With identity pose and zero translation, camera center = origin."""
        scene = load_colmap(binary_scene_dir)
        center = scene.images[1].camera_center
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-6)

    def test_c2w_shape(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.images[1].c2w.shape == (4, 4)
        assert scene.images[1].w2c.shape == (4, 4)

    def test_c2w_w2c_inverse(self, binary_scene_dir):
        """c2w and w2c should be inverses of each other."""
        scene = load_colmap(binary_scene_dir)
        img = scene.images[2]
        product = img.c2w @ img.w2c
        np.testing.assert_allclose(product, np.eye(4), atol=1e-6)

    def test_keypoints_shape(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        img = scene.images[1]
        assert img.xys.shape == (2, 2)
        assert img.point3D_ids.shape == (2,)

    def test_text_and_binary_agree(self, binary_scene_dir, text_scene_dir):
        """Both formats should produce identical poses."""
        bin_scene = load_colmap(binary_scene_dir)
        txt_scene = load_colmap(text_scene_dir)
        for img_id in [1, 2]:
            np.testing.assert_allclose(
                bin_scene.images[img_id].qvec,
                txt_scene.images[img_id].qvec,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                bin_scene.images[img_id].tvec,
                txt_scene.images[img_id].tvec,
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Point3D tests
# ---------------------------------------------------------------------------

class TestPoint3D:
    def test_point_count(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.num_points == 2

    def test_point_xyz(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        pt = scene.points3D[1]
        np.testing.assert_allclose(pt.xyz, [1.0, 2.0, 5.0])

    def test_point_rgb(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        pt = scene.points3D[1]
        np.testing.assert_array_equal(pt.rgb, [255, 128, 0])

    def test_point_error(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        assert scene.points3D[1].error == pytest.approx(0.5)

    def test_point_cloud_shape(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir)
        xyz, rgb = scene.point_cloud()
        assert xyz.shape == (2, 3)
        assert rgb.shape == (2, 3)

    def test_load_without_points(self, binary_scene_dir):
        scene = load_colmap(binary_scene_dir, load_points=False)
        assert scene.num_points == 0
        assert scene.num_images == 2  # images still loaded


# ---------------------------------------------------------------------------
# Math utility tests
# ---------------------------------------------------------------------------

class TestMathUtils:
    def test_identity_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = qvec_to_rotmat(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_qvec_rotmat_roundtrip(self):
        """Converting qvec -> R -> qvec should recover original quaternion."""
        q_orig = np.array([0.9239, 0.3827, 0.0, 0.0])
        q_orig /= np.linalg.norm(q_orig)
        R = qvec_to_rotmat(q_orig)
        q_recovered = rotmat_to_qvec(R)
        # Quaternions q and -q represent the same rotation
        assert (np.allclose(q_orig, q_recovered, atol=1e-6) or
                np.allclose(q_orig, -q_recovered, atol=1e-6))

    def test_rotation_matrix_is_orthogonal(self):
        q = np.array([0.7071, 0.7071, 0.0, 0.0])
        R = qvec_to_rotmat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            load_colmap("/nonexistent/path")

    def test_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_colmap(tmp_path)