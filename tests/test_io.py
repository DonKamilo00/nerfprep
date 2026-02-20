"""
tests/test_io.py
================
Tests for transforms.json, cameras.json, and PLY read/write.
Run with: pytest tests/test_io.py -v
"""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nerfprep.io import (
    load_cameras_json,
    load_ply,
    load_transforms_json,
    save_cameras_json,
    save_ply,
    save_transforms_json,
)


# ---------------------------------------------------------------------------
# Minimal mock ColmapScene for testing
# ---------------------------------------------------------------------------

class MockCamera:
    def __init__(self):
        self.id     = 1
        self.model  = "PINHOLE"
        self.width  = 1920
        self.height = 1080
        self.fx     = 800.0
        self.fy     = 800.0
        self.cx     = 960.0
        self.cy     = 540.0


class MockImage:
    def __init__(self, img_id, name, camera_id=1):
        self.id        = img_id
        self.name      = name
        self.camera_id = camera_id
        # Identity pose
        self._c2w = np.eye(4, dtype=np.float64)
        self._c2w[:3, 3] = [float(img_id), 0.0, 0.0]

    @property
    def c2w(self):
        return self._c2w


class MockScene:
    def __init__(self, n_images=3):
        self.cameras = {1: MockCamera()}
        self.images  = {
            i: MockImage(i, f"frame_{i:03d}.jpg")
            for i in range(1, n_images + 1)
        }

    def point_cloud(self):
        rng = np.random.default_rng(42)
        xyz = rng.random((10, 3)).astype(np.float64)
        rgb = (rng.random((10, 3)) * 255).astype(np.uint8)
        return xyz, rgb


@pytest.fixture
def scene():
    return MockScene(n_images=3)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# transforms.json
# ---------------------------------------------------------------------------

class TestTransformsJson:
    def test_saves_file(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        assert path.exists()

    def test_valid_json(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_required_keys(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        for key in ("fl_x", "fl_y", "cx", "cy", "w", "h", "frames"):
            assert key in data, f"Missing key: {key}"

    def test_intrinsics_correct(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        assert data["fl_x"] == pytest.approx(800.0)
        assert data["fl_y"] == pytest.approx(800.0)
        assert data["cx"]   == pytest.approx(960.0)
        assert data["cy"]   == pytest.approx(540.0)
        assert data["w"]    == 1920
        assert data["h"]    == 1080

    def test_frame_count(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["frames"]) == 3

    def test_frame_has_required_keys(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        frame = data["frames"][0]
        assert "file_path" in frame
        assert "transform_matrix" in frame

    def test_transform_matrix_shape(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        mat = data["frames"][0]["transform_matrix"]
        assert len(mat) == 4
        assert all(len(row) == 4 for row in mat)

    def test_file_path_includes_image_dir(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path, image_dir="images")
        with open(path) as f:
            data = json.load(f)
        assert data["frames"][0]["file_path"].startswith("images/")

    def test_opengl_convention_applied(self, scene, tmp):
        """Saved matrices should be in OpenGL convention (Y and Z flipped)."""
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        mat = np.array(data["frames"][0]["transform_matrix"])
        # Identity c2w in OpenCV → flip Y and Z rows in rotation block
        # For identity pose: R=I, t=[1,0,0]
        # After OpenCV→OpenGL: R[1] and R[2] should be negated
        assert mat[1, 1] == pytest.approx(-1.0, abs=1e-6)
        assert mat[2, 2] == pytest.approx(-1.0, abs=1e-6)

    def test_aabb_scale_stored(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path, aabb_scale=32)
        with open(path) as f:
            data = json.load(f)
        assert data["aabb_scale"] == 32

    def test_load_roundtrip(self, scene, tmp):
        path = tmp / "transforms.json"
        save_transforms_json(scene, path)
        loaded = load_transforms_json(path)
        assert "frames" in loaded
        assert len(loaded["frames"]) == 3
        assert isinstance(loaded["frames"][0]["transform_matrix"], np.ndarray)
        assert loaded["frames"][0]["transform_matrix"].shape == (4, 4)

    def test_load_missing_file_raises(self, tmp):
        with pytest.raises(FileNotFoundError):
            load_transforms_json(tmp / "nonexistent.json")


# ---------------------------------------------------------------------------
# cameras.json
# ---------------------------------------------------------------------------

class TestCamerasJson:
    def test_saves_file(self, scene, tmp):
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        assert path.exists()

    def test_valid_json(self, scene, tmp):
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_frame_count(self, scene, tmp):
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["frames"]) == 3

    def test_per_frame_intrinsics(self, scene, tmp):
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        frame = data["frames"][0]
        for key in ("fl_x", "fl_y", "cx", "cy", "w", "h"):
            assert key in frame, f"Missing per-frame key: {key}"

    def test_opencv_convention_preserved(self, scene, tmp):
        """cameras.json stores poses in OpenCV convention — no flip."""
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        with open(path) as f:
            data = json.load(f)
        mat = np.array(data["frames"][0]["transform_matrix"])
        # First image has identity rotation — R block should be identity
        np.testing.assert_allclose(mat[:3, :3], np.eye(3), atol=1e-8)

    def test_load_roundtrip(self, scene, tmp):
        path = tmp / "cameras.json"
        save_cameras_json(scene, path)
        loaded = load_cameras_json(path)
        assert len(loaded["frames"]) == 3
        mat = loaded["frames"][0]["transform_matrix"]
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (4, 4)

    def test_load_missing_file_raises(self, tmp):
        with pytest.raises(FileNotFoundError):
            load_cameras_json(tmp / "nonexistent.json")

    def test_transforms_and_cameras_json_agree_on_intrinsics(self, scene, tmp):
        save_transforms_json(scene, tmp / "t.json")
        save_cameras_json(scene,    tmp / "c.json")
        with open(tmp / "t.json") as f:
            t = json.load(f)
        with open(tmp / "c.json") as f:
            c = json.load(f)
        assert t["fl_x"] == pytest.approx(c["frames"][0]["fl_x"])
        assert t["cx"]   == pytest.approx(c["frames"][0]["cx"])


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

class TestPLY:
    def test_save_binary_creates_file(self, tmp):
        xyz = np.random.rand(10, 3).astype(np.float64)
        rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
        path = tmp / "test.ply"
        save_ply(xyz, rgb, path, binary=True)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_ascii_creates_file(self, tmp):
        xyz = np.random.rand(10, 3).astype(np.float64)
        path = tmp / "test.ply"
        save_ply(xyz, output_path=path, binary=False)
        assert path.exists()
        content = path.read_text()
        assert "ply" in content
        assert "ascii" in content

    def test_save_without_color(self, tmp):
        xyz  = np.random.rand(5, 3).astype(np.float64)
        path = tmp / "no_color.ply"
        save_ply(xyz, rgb=None, output_path=path)
        assert path.exists()

    def test_wrong_xyz_shape_raises(self, tmp):
        with pytest.raises(ValueError, match="xyz must be"):
            save_ply(np.zeros((10, 4)), output_path=tmp / "x.ply")

    def test_rgb_shape_mismatch_raises(self, tmp):
        xyz = np.zeros((10, 3))
        rgb = np.zeros((5, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="rgb must be"):
            save_ply(xyz, rgb, tmp / "x.ply")

    def test_binary_roundtrip_xyz(self, tmp):
        rng = np.random.default_rng(7)
        xyz = rng.random((50, 3)).astype(np.float64)
        path = tmp / "test.ply"
        save_ply(xyz, output_path=path, binary=True)
        xyz_loaded, rgb_loaded = load_ply(path)
        np.testing.assert_allclose(xyz_loaded, xyz, atol=1e-5)
        assert rgb_loaded is None

    def test_binary_roundtrip_with_color(self, tmp):
        rng = np.random.default_rng(8)
        xyz = rng.random((50, 3)).astype(np.float64)
        rgb = (rng.random((50, 3)) * 255).astype(np.uint8)
        path = tmp / "test.ply"
        save_ply(xyz, rgb, path, binary=True)
        xyz_l, rgb_l = load_ply(path)
        np.testing.assert_allclose(xyz_l, xyz, atol=1e-5)
        np.testing.assert_array_equal(rgb_l, rgb)

    def test_ascii_roundtrip_with_color(self, tmp):
        rng = np.random.default_rng(9)
        xyz = rng.random((20, 3)).astype(np.float64)
        rgb = (rng.random((20, 3)) * 255).astype(np.uint8)
        path = tmp / "test_ascii.ply"
        save_ply(xyz, rgb, path, binary=False)
        xyz_l, rgb_l = load_ply(path)
        np.testing.assert_allclose(xyz_l, xyz, atol=1e-5)
        np.testing.assert_array_equal(rgb_l, rgb)

    def test_float_rgb_auto_converted(self, tmp):
        """Float RGB in [0,1] should be auto-converted to uint8."""
        xyz = np.zeros((5, 3))
        rgb = np.ones((5, 3), dtype=np.float32)  # all 1.0 = white
        path = tmp / "float_rgb.ply"
        save_ply(xyz, rgb, path)
        _, rgb_l = load_ply(path)
        assert rgb_l is not None
        np.testing.assert_array_equal(rgb_l, np.full((5, 3), 255, dtype=np.uint8))

    def test_load_missing_file_raises(self, tmp):
        with pytest.raises(FileNotFoundError):
            load_ply(tmp / "nonexistent.ply")

    def test_point_count_preserved(self, tmp):
        N   = 123
        xyz = np.random.rand(N, 3)
        path = tmp / "count.ply"
        save_ply(xyz, output_path=path)
        xyz_l, _ = load_ply(path)
        assert len(xyz_l) == N

    def test_colmap_scene_roundtrip(self, scene, tmp):
        """Full pipeline: COLMAP scene → save_ply → load_ply."""
        xyz, rgb = scene.point_cloud()
        path = tmp / "scene.ply"
        save_ply(xyz, rgb, path)
        xyz_l, rgb_l = load_ply(path)
        np.testing.assert_allclose(xyz_l, xyz, atol=1e-5)
        np.testing.assert_array_equal(rgb_l, rgb)
