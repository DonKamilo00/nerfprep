"""
nerfprep.metrics
================
Standardized image quality metrics for neural rendering evaluation.

Every metric documents exactly which implementation it matches so your
numbers are reproducible and comparable across papers.

Metrics
-------
- psnr  : Peak Signal-to-Noise Ratio (dB). Higher is better.
- ssim  : Structural Similarity Index. Higher is better. [0, 1]
- lpips : Learned Perceptual Image Patch Similarity. Lower is better.
- evaluate : Compute all metrics at once.

All functions accept images as:
  - numpy arrays of shape (H, W, 3)
  - float32 or float64, pixel values in [0, 1]
  - OR uint8, pixel values in [0, 255] (auto-converted)

References
----------
PSNR   : Standard definition, matches most NeRF paper implementations.
SSIM   : Wang et al., "Image quality assessment: from error visibility to
         structural similarity", IEEE TIP 2004.
         Implementation matches skimage.metrics.structural_similarity
         with win_size=11, gaussian_weights=True, sigma=1.5, K1=0.01, K2=0.03.
LPIPS  : Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
         Perceptual Metric", CVPR 2018. Uses VGG backbone, matches the
         official implementation at https://github.com/richzhang/PerceptualSimilarity

Usage
-----
    import numpy as np
    from nerfprep.metrics import evaluate, psnr, ssim

    render = np.random.rand(800, 600, 3).astype(np.float32)
    gt     = np.random.rand(800, 600, 3).astype(np.float32)

    # All metrics at once
    results = evaluate(render, gt)
    print(results)
    # {'psnr': 8.23, 'ssim': 0.012, 'lpips': 0.71}

    # Individual metrics
    p = psnr(render, gt)
    s = ssim(render, gt)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_normalize(
    img: np.ndarray,
    name: str = "image",
) -> np.ndarray:
    """
    Validate image shape and dtype, normalize to float64 in [0, 1].
    Accepts (H, W, 3) float in [0,1] or uint8 in [0, 255].
    """
    img = np.asarray(img)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (H, W, 3), got {img.shape}"
        )

    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0

    img = img.astype(np.float64)

    if img.max() > 1.0 + 1e-6:
        warnings.warn(
            f"{name} has values > 1.0 (max={img.max():.3f}). "
            "Expected float images in [0, 1]. "
            "Pass uint8 images in [0, 255] for auto-conversion.",
            UserWarning,
            stacklevel=3,
        )

    return np.clip(img, 0.0, 1.0)


def _check_same_shape(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape. "
            f"Got pred={pred.shape}, gt={gt.shape}"
        )


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB). Higher is better.

    Computed as: 10 * log10(max_val² / MSE)

    Matches the standard implementation used by NeRF, mip-NeRF, 3DGS and
    most neural rendering papers. When MSE=0 (perfect reconstruction),
    returns inf.

    Args:
        pred:    (H, W, 3) predicted/rendered image
        gt:      (H, W, 3) ground truth image
        max_val: Maximum pixel value (1.0 for float, 255.0 for uint8).
                 Auto-handled if you pass uint8 arrays.

    Returns:
        PSNR value in dB (float). Returns inf for perfect reconstruction.

    Example:
        p = psnr(render, gt_image)
        print(f"PSNR: {p:.2f} dB")
    """
    pred = _validate_and_normalize(pred, "pred")
    gt   = _validate_and_normalize(gt,   "gt")
    _check_same_shape(pred, gt)

    mse = np.mean((pred - gt) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """1D Gaussian kernel, normalized to sum to 1."""
    x      = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_filter_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a separable 2D Gaussian filter to a (H, W) image.
    Equivalent to scipy.ndimage.convolve1d applied along each axis.
    """
    from numpy import convolve
    H, W = img.shape
    # Filter along columns (axis=1) then rows (axis=0)
    out = np.zeros_like(img)
    pad = len(kernel) // 2
    for i in range(H):
        out[i] = np.convolve(img[i], kernel, mode="same")
    result = np.zeros_like(img)
    for j in range(W):
        result[:, j] = np.convolve(out[:, j], kernel, mode="same")
    return result


def ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    pred = _validate_and_normalize(pred, "pred")
    gt   = _validate_and_normalize(gt,   "gt")
    _check_same_shape(pred, gt)

    # Clamp win_size to image dimensions (must be odd)
    min_dim  = min(pred.shape[0], pred.shape[1])
    win_size = min(win_size, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    kernel = _gaussian_kernel_1d(win_size, sigma)
    """
    Structural Similarity Index (SSIM). Higher is better. Range: [-1, 1].

    Matches skimage.metrics.structural_similarity with:
        win_size=11, gaussian_weights=True, sigma=1.5,
        K1=0.01, K2=0.03, multichannel=True

    This is the standard configuration used in NeRF, mip-NeRF 360,
    and 3D Gaussian Splatting papers.

    Args:
        pred:       (H, W, 3) predicted/rendered image
        gt:         (H, W, 3) ground truth image
        win_size:   Gaussian window size (default 11, matches papers)
        sigma:      Gaussian standard deviation (default 1.5)
        K1, K2:     Stability constants (default 0.01, 0.03)
        data_range: Value range of images (1.0 for float [0,1])

    Returns:
        Mean SSIM over all channels (float in [-1, 1], typically [0, 1])

    Example:
        s = ssim(render, gt_image)
        print(f"SSIM: {s:.4f}")
    """
    pred = _validate_and_normalize(pred, "pred")
    gt   = _validate_and_normalize(gt,   "gt")
    _check_same_shape(pred, gt)

    kernel = _gaussian_kernel_1d(win_size, sigma)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    channel_ssims = []
    for c in range(3):
        p = pred[:, :, c]
        g = gt[:, :, c]

        mu_p  = _gaussian_filter_2d(p, kernel)
        mu_g  = _gaussian_filter_2d(g, kernel)
        mu_pp = _gaussian_filter_2d(p * p, kernel)
        mu_gg = _gaussian_filter_2d(g * g, kernel)
        mu_pg = _gaussian_filter_2d(p * g, kernel)

        sigma_p  = mu_pp - mu_p ** 2
        sigma_g  = mu_gg - mu_g ** 2
        sigma_pg = mu_pg - mu_p * mu_g

        numerator   = (2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)
        denominator = (mu_p**2 + mu_g**2 + C1) * (sigma_p + sigma_g + C2)

        ssim_map = numerator / denominator
        channel_ssims.append(float(ssim_map.mean()))

    return float(np.mean(channel_ssims))


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    net: str = "vgg",
) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS). Lower is better.

    Matches the official implementation by Zhang et al. (CVPR 2018).
    Uses VGG backbone by default, which is the standard for NeRF papers.
    AlexNet backbone is available but less commonly reported.

    Requires: pip install "nerfprep[metrics]"
    (installs torch and lpips package)

    Args:
        pred: (H, W, 3) predicted/rendered image
        gt:   (H, W, 3) ground truth image
        net:  Backbone network. 'vgg' (default, matches most papers)
              or 'alex' (faster, slightly different scores)

    Returns:
        LPIPS distance (float, lower = more similar). Typically in [0, 1].

    Example:
        l = lpips(render, gt_image)
        print(f"LPIPS: {l:.4f}")
    """
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        raise ImportError(
            "LPIPS requires torch and the lpips package. Install with:\n"
            '    pip install "nerfprep[metrics]"\n'
            "or:\n"
            "    pip install torch lpips"
        )

    pred = _validate_and_normalize(pred, "pred")
    gt   = _validate_and_normalize(gt,   "gt")
    _check_same_shape(pred, gt)

    # Cache the loss function to avoid reloading weights on every call
    if not hasattr(lpips, "_loss_fn_cache"):
        lpips._loss_fn_cache = {}
    if net not in lpips._loss_fn_cache:
        lpips._loss_fn_cache[net] = lpips_lib.LPIPS(net=net)
    loss_fn = lpips._loss_fn_cache[net]

    # Convert (H, W, 3) float64 numpy → (1, 3, H, W) float32 tensor in [-1, 1]
    def to_tensor(img: np.ndarray) -> "torch.Tensor":
        img_t = torch.from_numpy(img.astype(np.float32))   # (H, W, 3)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)        # (1, 3, H, W)
        return img_t * 2.0 - 1.0                           # [0,1] → [-1,1]

    with torch.no_grad():
        dist = loss_fn(to_tensor(pred), to_tensor(gt))

    return float(dist.item())


# ---------------------------------------------------------------------------
# evaluate — all metrics at once
# ---------------------------------------------------------------------------

def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    compute_lpips: bool = False,
    lpips_net: str = "vgg",
) -> Dict[str, float]:
    """
    Compute all image quality metrics between a render and ground truth.

    By default computes PSNR and SSIM (no extra dependencies).
    Set compute_lpips=True to also compute LPIPS (requires torch + lpips).

    Args:
        pred:          (H, W, 3) predicted/rendered image
        gt:            (H, W, 3) ground truth image
        compute_lpips: Whether to compute LPIPS (default False)
        lpips_net:     LPIPS backbone ('vgg' or 'alex')

    Returns:
        Dict with keys 'psnr', 'ssim', and optionally 'lpips'.
        All values are floats.

    Example:
        results = evaluate(render, gt_image)
        print(results)
        # {'psnr': 32.14, 'ssim': 0.921}

        results = evaluate(render, gt_image, compute_lpips=True)
        print(results)
        # {'psnr': 32.14, 'ssim': 0.921, 'lpips': 0.087}
    """
    results: Dict[str, float] = {
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
    }
    if compute_lpips:
        results["lpips"] = lpips(pred, gt, net=lpips_net)

    return results


# ---------------------------------------------------------------------------
# evaluate_dataset — average metrics over a list of image pairs
# ---------------------------------------------------------------------------

def evaluate_dataset(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    compute_lpips: bool = False,
    lpips_net: str = "vgg",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute mean metrics over a list of rendered and ground truth images.

    This is the function to call at the end of your evaluation loop — it
    matches the reporting convention used in NeRF, mip-NeRF 360, and 3DGS
    papers (mean over all test views).

    Args:
        preds:         List of (H, W, 3) predicted images
        gts:           List of (H, W, 3) ground truth images
        compute_lpips: Whether to compute LPIPS
        lpips_net:     LPIPS backbone
        verbose:       If True, prints per-image metrics

    Returns:
        Dict with mean 'psnr', 'ssim', and optionally 'lpips'.

    Example:
        renders = [render_view(cam) for cam in test_cameras]
        gt_imgs = [load_image(p) for p in test_image_paths]

        results = evaluate_dataset(renders, gt_imgs)
        print(f"PSNR: {results['psnr']:.2f} dB")
        print(f"SSIM: {results['ssim']:.4f}")
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"Number of predictions ({len(preds)}) must match "
            f"ground truth ({len(gts)})"
        )
    if len(preds) == 0:
        raise ValueError("Empty list of images provided.")

    all_results = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        result = evaluate(pred, gt, compute_lpips=compute_lpips, lpips_net=lpips_net)
        all_results.append(result)
        if verbose:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in result.items())
            print(f"  [{i+1:3d}/{len(preds)}] {parts}")

    # Average over all views
    keys = all_results[0].keys()
    return {k: float(np.mean([r[k] for r in all_results])) for k in keys}
