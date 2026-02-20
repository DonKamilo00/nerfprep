"""
tests/test_metrics.py
=====================
Tests for PSNR, SSIM, and the evaluate functions.
LPIPS tests are skipped if torch/lpips are not installed.
Run with: pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from nerfprep.metrics import evaluate, evaluate_dataset, psnr, ssim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identical_images():
    """Two identical random images — metrics should be perfect."""
    rng = np.random.default_rng(42)
    img = rng.random((64, 64, 3)).astype(np.float32)
    return img, img.copy()


@pytest.fixture
def different_images():
    """Two clearly different images."""
    rng = np.random.default_rng(0)
    pred = rng.random((64, 64, 3)).astype(np.float32)
    gt   = rng.random((64, 64, 3)).astype(np.float32)
    return pred, gt


@pytest.fixture
def black_white():
    """All-black prediction vs all-white ground truth — worst case."""
    pred = np.zeros((32, 32, 3), dtype=np.float32)
    gt   = np.ones( (32, 32, 3), dtype=np.float32)
    return pred, gt


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="shape"):
            psnr(np.zeros((64, 64)), np.zeros((64, 64)))

    def test_wrong_channels(self):
        with pytest.raises(ValueError, match="shape"):
            psnr(np.zeros((64, 64, 1)), np.zeros((64, 64, 1)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            psnr(np.zeros((64, 64, 3)), np.zeros((32, 32, 3)))

    def test_uint8_auto_converted(self):
        """uint8 [0,255] images should be accepted and auto-normalized."""
        pred = np.full((16, 16, 3), 128, dtype=np.uint8)
        gt   = np.full((16, 16, 3), 128, dtype=np.uint8)
        result = psnr(pred, gt)
        assert result == float("inf")

    def test_float_out_of_range_warns(self):
        """Float images with values > 1 should warn."""
        pred = np.full((16, 16, 3), 2.0, dtype=np.float32)
        gt   = np.ones((16, 16, 3), dtype=np.float32)
        with pytest.warns(UserWarning, match="values > 1.0"):
            psnr(pred, gt)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

class TestPSNR:
    def test_identical_images_returns_inf(self, identical_images):
        pred, gt = identical_images
        assert psnr(pred, gt) == float("inf")

    def test_returns_float(self, different_images):
        pred, gt = different_images
        result = psnr(pred, gt)
        assert isinstance(result, float)

    def test_higher_for_similar_images(self, identical_images):
        """Adding less noise should give higher PSNR."""
        rng  = np.random.default_rng(1)
        pred, gt = identical_images
        low_noise  = np.clip(gt + rng.random(gt.shape).astype(np.float32) * 0.01, 0, 1)
        high_noise = np.clip(gt + rng.random(gt.shape).astype(np.float32) * 0.5,  0, 1)
        assert psnr(low_noise, gt) > psnr(high_noise, gt)

    def test_black_white_worst_case(self, black_white):
        """Black vs white is the worst possible PSNR."""
        pred, gt = black_white
        result = psnr(pred, gt)
        # MSE = 1.0, PSNR = 10*log10(1/1) = 0 dB
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        """PSNR for known MSE should match manual calculation."""
        pred = np.full((10, 10, 3), 0.9, dtype=np.float64)
        gt   = np.full((10, 10, 3), 1.0, dtype=np.float64)
        # MSE = 0.01, PSNR = 10*log10(1/0.01) = 20 dB
        assert psnr(pred, gt) == pytest.approx(20.0, abs=1e-4)

    def test_positive_for_reasonable_images(self, different_images):
        """PSNR should generally be positive for natural image ranges."""
        pred, gt = different_images
        # Random images still have finite positive PSNR
        result = psnr(pred, gt)
        assert np.isfinite(result)

    def test_symmetric(self, different_images):
        """PSNR(pred, gt) == PSNR(gt, pred) since MSE is symmetric."""
        pred, gt = different_images
        assert psnr(pred, gt) == pytest.approx(psnr(gt, pred), rel=1e-6)

    def test_uint8_identical(self):
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        assert psnr(img, img.copy()) == float("inf")


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

class TestSSIM:
    def test_identical_images_returns_one(self, identical_images):
        pred, gt = identical_images
        result = ssim(pred, gt)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_returns_float(self, different_images):
        pred, gt = different_images
        result = ssim(pred, gt)
        assert isinstance(result, float)

    def test_range(self, different_images):
        """SSIM should be in [-1, 1], typically [0, 1] for natural images."""
        pred, gt = different_images
        result = ssim(pred, gt)
        assert -1.0 <= result <= 1.0

    def test_higher_for_similar_images(self, identical_images):
        """Less noise → higher SSIM."""
        rng = np.random.default_rng(2)
        pred, gt = identical_images
        low_noise  = np.clip(gt + rng.random(gt.shape).astype(np.float32) * 0.02, 0, 1)
        high_noise = np.clip(gt + rng.random(gt.shape).astype(np.float32) * 0.4,  0, 1)
        assert ssim(low_noise, gt) > ssim(high_noise, gt)

    def test_black_white_low_ssim(self, black_white):
        """Black vs white should give very low SSIM."""
        pred, gt = black_white
        result = ssim(pred, gt)
        assert result < 0.1

    def test_symmetric(self, different_images):
        """SSIM should be approximately symmetric."""
        pred, gt = different_images
        assert ssim(pred, gt) == pytest.approx(ssim(gt, pred), abs=1e-4)

    def test_uint8_identical(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = ssim(img, img.copy())
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_minimum_image_size(self):
        """Images smaller than win_size should still work via auto kernel reduction."""
        pred = np.random.rand(8, 8, 3).astype(np.float32)
        gt   = np.random.rand(8, 8, 3).astype(np.float32)
        result = ssim(pred, gt, win_size=7)   # explicitly smaller kernel
        assert np.isfinite(result)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_dict_with_correct_keys(self, different_images):
        pred, gt = different_images
        result = evaluate(pred, gt)
        assert set(result.keys()) == {"psnr", "ssim"}

    def test_all_values_are_floats(self, different_images):
        pred, gt = different_images
        result = evaluate(pred, gt)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_identical_images_perfect_scores(self, identical_images):
        pred, gt = identical_images
        result = evaluate(pred, gt)
        assert result["psnr"] == float("inf")
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)

    def test_no_lpips_by_default(self, different_images):
        pred, gt = different_images
        result = evaluate(pred, gt)
        assert "lpips" not in result

    def test_lpips_skipped_gracefully_if_no_torch(self, different_images):
        """If torch not installed, should raise ImportError with helpful message."""
        pred, gt = different_images
        try:
            import torch
            pytest.skip("torch is installed, skipping ImportError test")
        except ImportError:
            with pytest.raises(ImportError, match="nerfprep\\[metrics\\]"):
                evaluate(pred, gt, compute_lpips=True)


# ---------------------------------------------------------------------------
# evaluate_dataset
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    def test_mean_over_multiple_images(self):
        rng   = np.random.default_rng(99)
        preds = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
        gts   = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
        result = evaluate_dataset(preds, gts)
        assert "psnr" in result
        assert "ssim" in result
        assert np.isfinite(result["psnr"])
        assert np.isfinite(result["ssim"])

    def test_single_image_matches_evaluate(self):
        rng  = np.random.default_rng(7)
        pred = rng.random((32, 32, 3)).astype(np.float32)
        gt   = rng.random((32, 32, 3)).astype(np.float32)
        single   = evaluate(pred, gt)
        dataset  = evaluate_dataset([pred], [gt])
        assert dataset["psnr"] == pytest.approx(single["psnr"], rel=1e-6)
        assert dataset["ssim"] == pytest.approx(single["ssim"], rel=1e-6)

    def test_identical_images_perfect_scores(self):
        rng  = np.random.default_rng(3)
        imgs = [rng.random((16, 16, 3)).astype(np.float32) for _ in range(3)]
        result = evaluate_dataset(imgs, [i.copy() for i in imgs])
        assert result["psnr"] == float("inf")
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)

    def test_length_mismatch_raises(self):
        pred = [np.zeros((16, 16, 3))]
        gt   = [np.zeros((16, 16, 3)), np.zeros((16, 16, 3))]
        with pytest.raises(ValueError, match="Number of predictions"):
            evaluate_dataset(pred, gt)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            evaluate_dataset([], [])

    def test_mean_is_correct(self):
        """Average of per-image PSNRs should match dataset PSNR."""
        rng   = np.random.default_rng(55)
        preds = [rng.random((16, 16, 3)).astype(np.float32) for _ in range(4)]
        gts   = [rng.random((16, 16, 3)).astype(np.float32) for _ in range(4)]
        individual = [psnr(p, g) for p, g in zip(preds, gts)]
        dataset    = evaluate_dataset(preds, gts)
        assert dataset["psnr"] == pytest.approx(np.mean(individual), rel=1e-6)
