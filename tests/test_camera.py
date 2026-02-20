"""
tests/test_camera.py
====================
Tests for PinholeCamera, convert_convention, and Rays.
Run with: pytest tests/test_camera.py -v
"""

import numpy as np
import pytest

from nerfprep.camera import PinholeCamera, Rays, convert_convention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cam():
    """Standard 1080p pinhole camera."""
    return PinholeCamera(fx=800.0, fy=800.0, cx=960.0, cy=540.0,
                         width=1920, height=1080)


@pytest.fixture
def identity_c2w():
    """Camera sitting at world origin, looking along +Z (OpenCV)."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def translated_c2w():
    """Camera at (1, 0, 0), looking along +Z."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.0, 0.0, 0.0]
    return c2w


# ---------------------------------------------------------------------------
# PinholeCamera construction
# ---------------------------------------------------------------------------

class TestPinholeCameraConstruction:
    def test_basic_construction(self, cam):
        assert cam.fx == 800.0
        assert cam.fy == 800.0
        assert cam.cx == 960.0
        assert cam.cy == 540.0
        assert cam.width == 1920
        assert cam.height == 1080

    def test_from_colmap(self):
        """Test construction from a mock COLMAP camera object."""
        class MockCamera:
            fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
            width, height  = 640, 480

        cam = PinholeCamera.from_colmap(MockCamera())
        assert cam.fx == 600.0
        assert cam.width == 640

    def test_from_fov_degrees(self):
        cam = PinholeCamera.from_fov(fov_x=90.0, width=800, height=600)
        # For 90 deg FOV: fx = (W/2) / tan(45 deg) = W/2
        assert cam.fx == pytest.approx(400.0, rel=1e-4)
        assert cam.cx == 400.0
        assert cam.cy == 300.0

    def test_from_fov_radians(self):
        cam_deg = PinholeCamera.from_fov(60.0, 800, 600, fov_in_degrees=True)
        cam_rad = PinholeCamera.from_fov(np.deg2rad(60.0), 800, 600, fov_in_degrees=False)
        assert cam_deg.fx == pytest.approx(cam_rad.fx, rel=1e-6)

    def test_K_matrix(self, cam):
        K = cam.K
        assert K.shape == (3, 3)
        assert K[0, 0] == pytest.approx(800.0)
        assert K[1, 1] == pytest.approx(800.0)
        assert K[0, 2] == pytest.approx(960.0)
        assert K[1, 2] == pytest.approx(540.0)
        assert K[2, 2] == pytest.approx(1.0)
        assert K[0, 1] == pytest.approx(0.0)   # no skew

    def test_K_inv(self, cam):
        np.testing.assert_allclose(cam.K @ cam.K_inv, np.eye(3), atol=1e-10)

    def test_fov_properties(self, cam):
        assert cam.fov_x == pytest.approx(2 * np.arctan2(960, 800), rel=1e-6)
        assert cam.fov_y == pytest.approx(2 * np.arctan2(540, 800), rel=1e-6)

    def test_aspect_ratio(self, cam):
        assert cam.aspect_ratio == pytest.approx(1920 / 1080, rel=1e-6)

    def test_repr(self, cam):
        r = repr(cam)
        assert "PinholeCamera" in r
        assert "800" in r

    def test_rescale_half(self, cam):
        small = cam.rescale(0.5)
        assert small.fx == pytest.approx(400.0)
        assert small.width == 960
        assert small.height == 540
        assert small.cx == pytest.approx(480.0)

    def test_rescale_2x(self, cam):
        big = cam.rescale(2.0)
        assert big.width == 3840
        assert big.fx == pytest.approx(1600.0)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_project_principal_axis(self, cam):
        """Point on principal axis should project to principal point."""
        pt = np.array([[0.0, 0.0, 1.0]])
        uv = cam.project(pt)
        assert uv[0, 0] == pytest.approx(cam.cx)
        assert uv[0, 1] == pytest.approx(cam.cy)

    def test_project_known_point(self, cam):
        """Point at (1, 1, 1) should project to (cx + fx, cy + fy)."""
        pt = np.array([[1.0, 1.0, 1.0]])
        uv = cam.project(pt)
        assert uv[0, 0] == pytest.approx(cam.cx + cam.fx)
        assert uv[0, 1] == pytest.approx(cam.cy + cam.fy)

    def test_project_depth_scaling(self, cam):
        """Projection should be invariant to depth scaling of direction."""
        pt1 = np.array([[1.0, 1.0, 2.0]])
        pt2 = np.array([[2.0, 2.0, 4.0]])
        np.testing.assert_allclose(cam.project(pt1), cam.project(pt2), atol=1e-10)

    def test_project_batch(self, cam):
        """Batch projection should work for (N, 3) inputs."""
        pts = np.random.randn(100, 3)
        pts[:, 2] = np.abs(pts[:, 2]) + 0.1   # ensure positive depth
        uv = cam.project(pts)
        assert uv.shape == (100, 2)

    def test_project_batch_2d(self, cam):
        """Should work for (H, W, 3) inputs too."""
        pts = np.ones((10, 20, 3))
        uv  = cam.project(pts)
        assert uv.shape == (10, 20, 2)


# ---------------------------------------------------------------------------
# Unprojection
# ---------------------------------------------------------------------------

class TestUnprojection:
    def test_unproject_full_depth_map(self, cam):
        depth = np.ones((cam.height, cam.width))
        pts   = cam.unproject(depth)
        assert pts.shape == (cam.height, cam.width, 3)

    def test_unproject_depth_at_principal_point(self, cam):
        """Unprojecting principal point pixel should give a ray along Z."""
        pixel = np.array([[cam.cx, cam.cy]])
        depth = np.array([2.0])
        pt    = cam.unproject(depth, pixels=pixel)
        # Should be (0, 0, 2) — on optical axis
        np.testing.assert_allclose(pt[0, :2], [0.0, 0.0], atol=1e-10)
        assert pt[0, 2] == pytest.approx(2.0)

    def test_project_unproject_roundtrip(self, cam):
        """Projecting a point and unprojecting should recover original."""
        pts_3d = np.array([[0.5, -0.3, 3.0], [1.0, 2.0, 5.0]])
        uvs    = cam.project(pts_3d)
        depths = pts_3d[:, 2]
        pts_recovered = cam.unproject(depths, pixels=uvs)
        np.testing.assert_allclose(pts_recovered, pts_3d, atol=1e-8)

    def test_unproject_specific_pixels(self, cam):
        pixels = np.array([[0.0, 0.0], [100.0, 200.0]])
        depths = np.array([1.0, 2.0])
        pts    = cam.unproject(depths, pixels=pixels)
        assert pts.shape == (2, 3)


# ---------------------------------------------------------------------------
# Frustum
# ---------------------------------------------------------------------------

class TestFrustum:
    def test_in_frustum_center(self, cam):
        """Point on principal axis in front of camera should be visible."""
        pt = np.array([[0.0, 0.0, 1.0]])
        assert cam.in_frustum(pt)[0] == True

    def test_behind_camera(self, cam):
        pt = np.array([[0.0, 0.0, -1.0]])
        assert cam.in_frustum(pt)[0] == False

    def test_out_of_bounds_x(self, cam):
        """Point far to the right, outside image width."""
        pt = np.array([[1e6, 0.0, 1.0]])
        assert cam.in_frustum(pt)[0] == False

    def test_batch_frustum(self, cam):
        pts = np.array([
            [0.0,  0.0, 1.0],   # in
            [0.0,  0.0, -1.0],  # behind
            [1e6,  0.0, 1.0],   # out of bounds
        ])
        mask = cam.in_frustum(pts)
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == False


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

class TestRayGeneration:
    def test_ray_shape_full_image(self, cam, identity_c2w):
        rays = cam.generate_rays(identity_c2w)
        assert rays.origins.shape    == (cam.height, cam.width, 3)
        assert rays.directions.shape == (cam.height, cam.width, 3)

    def test_ray_origins_at_camera_center(self, cam, translated_c2w):
        """All rays should originate from the camera center."""
        rays = cam.generate_rays(translated_c2w)
        expected_origin = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rays.origins[0, 0], expected_origin, atol=1e-10)
        np.testing.assert_allclose(rays.origins[-1, -1], expected_origin, atol=1e-10)

    def test_ray_directions_normalized(self, cam, identity_c2w):
        """Ray directions should be unit vectors when normalize=True."""
        rays  = cam.generate_rays(identity_c2w, normalize=True)
        norms = np.linalg.norm(rays.directions, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)

    def test_center_ray_along_z(self, cam, identity_c2w):
        """With identity pose, the center ray should point along +Z."""
        rays     = cam.generate_rays(identity_c2w)
        cy_idx   = int(cam.cy)
        cx_idx   = int(cam.cx)
        center   = rays.directions[cy_idx, cx_idx]
        np.testing.assert_allclose(center, [0.0, 0.0, 1.0], atol=1e-3)

    def test_ray_generation_specific_pixels(self, cam, identity_c2w):
        """Generating rays for specific pixels should give (N, 3) output."""
        pixels = np.array([[0.0, 0.0], [960.0, 540.0], [1919.0, 1079.0]])
        rays   = cam.generate_rays(identity_c2w, pixels=pixels)
        assert rays.origins.shape    == (3, 3)
        assert rays.directions.shape == (3, 3)

    def test_specific_pixels_normalized(self, cam, identity_c2w):
        pixels = np.random.uniform(0, [cam.width, cam.height], size=(50, 2))
        rays   = cam.generate_rays(identity_c2w, pixels=pixels)
        norms  = np.linalg.norm(rays.directions, axis=-1)
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-6)

    def test_invalid_c2w_shape(self, cam):
        with pytest.raises(ValueError):
            cam.generate_rays(np.eye(3))

    def test_rays_at(self, cam, identity_c2w):
        """Rays.at(t) should return correct positions."""
        pixels = np.array([[cam.cx, cam.cy]])
        rays   = cam.generate_rays(identity_c2w, pixels=pixels)
        t      = np.array([5.0])
        pos    = rays.at(t)
        # Center ray points along +Z, so at t=5 we should be at (0, 0, 5)
        np.testing.assert_allclose(pos[0], [0.0, 0.0, 5.0], atol=1e-3)

    def test_rays_reshape(self, cam, identity_c2w):
        rays    = cam.generate_rays(identity_c2w)                   # (H, W, 3)
        flat    = rays.reshape(cam.height * cam.width)              # (H*W, 3)
        assert flat.origins.shape == (cam.height * cam.width, 3)


# ---------------------------------------------------------------------------
# Pixel grid
# ---------------------------------------------------------------------------

class TestPixelGrid:
    def test_pixel_grid_shape(self, cam):
        grid = cam.pixel_grid()
        assert grid.shape == (cam.height, cam.width, 2)

    def test_pixel_grid_corners(self, cam):
        grid = cam.pixel_grid()
        np.testing.assert_allclose(grid[0, 0],  [0.0, 0.0])
        np.testing.assert_allclose(grid[0, -1], [cam.width - 1, 0.0])
        np.testing.assert_allclose(grid[-1, 0], [0.0, cam.height - 1])


# ---------------------------------------------------------------------------
# Convention conversion
# ---------------------------------------------------------------------------

class TestConventionConversion:
    def test_identity_same_convention(self):
        c2w = np.eye(4)
        result = convert_convention(c2w, src="opencv", dst="opencv")
        np.testing.assert_allclose(result, c2w)

    def test_colmap_same_as_opencv(self):
        """COLMAP and OpenCV conventions are identical."""
        c2w = np.eye(4)
        result = convert_convention(c2w, src="colmap", dst="opencv")
        np.testing.assert_allclose(result, c2w, atol=1e-10)

    def test_opencv_opengl_roundtrip(self):
        """Converting OpenCV → OpenGL → OpenCV should recover original."""
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        to_gl   = convert_convention(c2w, src="opencv", dst="opengl")
        back    = convert_convention(to_gl, src="opengl", dst="opencv")
        np.testing.assert_allclose(back, c2w, atol=1e-10)

    def test_all_roundtrips(self):
        """Any src→dst→src roundtrip should recover the original."""
        conventions = ["opencv", "opengl", "blender", "colmap"]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [0.5, -1.0, 2.0]
        for src in conventions:
            for dst in conventions:
                to_dst = convert_convention(c2w, src=src, dst=dst)
                back   = convert_convention(to_dst, src=dst, dst=src)
                np.testing.assert_allclose(back, c2w, atol=1e-10,
                    err_msg=f"Roundtrip failed: {src}→{dst}→{src}")

    def test_opencv_to_opengl_flips_y_z(self):
        """OpenCV→OpenGL should negate Y and Z components of translation."""
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        result = convert_convention(c2w, src="opencv", dst="opengl")
        # Translation Y and Z should flip
        assert result[1, 3] == pytest.approx(-2.0, abs=1e-9)
        assert result[2, 3] == pytest.approx(-3.0, abs=1e-9)
        assert result[0, 3] == pytest.approx( 1.0, abs=1e-9)

    def test_invalid_convention(self):
        with pytest.raises(ValueError):
            convert_convention(np.eye(4), src="invalid", dst="opencv")

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            convert_convention(np.eye(3), src="opencv", dst="opengl")

    def test_rotation_remains_orthogonal(self):
        """After conversion the rotation block should still be orthogonal."""
        angle = np.pi / 6
        c2w   = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        result = convert_convention(c2w, src="opencv", dst="opengl")
        R = result[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Rays dataclass
# ---------------------------------------------------------------------------

class TestRays:
    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            Rays(origins=np.zeros((3, 3)), directions=np.zeros((4, 3)))

    def test_len(self):
        rays = Rays(origins=np.zeros((10, 3)), directions=np.zeros((10, 3)))
        assert len(rays) == 10

    def test_at_scalar(self):
        origins    = np.array([[0.0, 0.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        rays = Rays(origins=origins, directions=directions)
        pos  = rays.at(np.array([3.0]))
        np.testing.assert_allclose(pos, [[0.0, 0.0, 3.0]])
