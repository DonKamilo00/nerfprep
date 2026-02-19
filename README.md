# nerfprep

**The missing standard library for neural rendering data preparation.**

Stop rewriting the same 500 lines of data plumbing every project. `nerfprep` gives you a clean, lightweight, framework-agnostic toolkit for everything that happens *before* you start training — COLMAP parsing, dataset loading, camera math, metric evaluation, and format conversion.

```python
import nerfprep as nrp

# Load a COLMAP reconstruction in one line
scene = nrp.load_colmap("data/scene/sparse/0")

# Clean camera intrinsics and poses, ready to use
K   = scene.cameras[1].K          # 3x3 intrinsic matrix
c2w = scene.images[1].c2w         # 4x4 camera-to-world matrix
xyz, rgb = scene.point_cloud()     # (N,3) point cloud arrays

# Evaluate renders against ground truth
metrics = nrp.evaluate(renders, gt_images)  # PSNR, SSIM, LPIPS
```

No PyTorch. No OpenCV. No heavy dependencies. Just NumPy and clean Python.

---

## Why nerfprep?

Every neural rendering project starts the same way:

- 🔁 Copy-paste the COLMAP parser from your last repo
- 📁 Spend half a day wrangling a new dataset's folder structure
- 🌐 Figure out whether this codebase uses OpenCV or OpenGL convention
- 📊 Argue with yourself about which SSIM implementation matches the paper

`nerfprep` solves these once, correctly, with tests — so you can spend your time on actual research.

---

## Installation

```bash
pip install nerfprep
```

For metric evaluation (requires PyTorch):

```bash
pip install "nerfprep[metrics]"
```

> **Note:** `nerfprep` is in early development. The API may change before v1.0. Pin your version in requirements.

---

## Modules

### `nerfprep.datasets` — Data loading

**COLMAP** (✅ available now)

Parses COLMAP sparse reconstructions. Supports both binary (`.bin`) and text (`.txt`) formats, auto-detected.

```python
from nerfprep.datasets.colmap import load_colmap

scene = load_colmap("sparse/0")

# Camera intrinsics
cam = scene.cameras[1]
print(cam.model)   # 'PINHOLE'
print(cam.K)       # 3x3 intrinsic matrix
print(cam.fx, cam.fy, cam.cx, cam.cy)

# Image poses — all in one place, no quaternion math required
img = scene.images[1]
print(img.R)               # 3x3 rotation matrix
print(img.c2w)             # 4x4 camera-to-world
print(img.w2c)             # 4x4 world-to-camera
print(img.camera_center)   # (3,) world-space camera position

# 3D point cloud
xyz, rgb = scene.point_cloud()   # (N,3) float64, (N,3) uint8

print(scene.summary())
# ColmapScene
#   cameras  : 1
#   images   : 312
#   points3D : 184,211
```

---

### `nerfprep.camera` — Camera math *(coming soon)*

All the camera utilities you keep rewriting, in one place.

```python
from nerfprep.camera import PinholeCamera, convert_convention

cam = PinholeCamera(fx=800, fy=800, cx=960, cy=540, width=1920, height=1080)

# Project 3D points to 2D
points_3d = np.array([[1.0, 0.5, 3.0]])
pixels = cam.project(points_3d)

# Unproject depth map to 3D
xyz = cam.unproject(depth_map)

# Coordinate convention conversion — no more guessing
R_opengl = convert_convention(R_opencv, src="opencv", dst="opengl")
```

Supported conventions: `opencv`, `opengl`, `colmap`, `blender`

---

### `nerfprep.datasets` — Dataset loaders *(coming soon)*

One consistent API across the most common benchmarks. No more custom dataloaders.

```python
from nerfprep.datasets import load_dataset

# Same API regardless of dataset
dataset = load_dataset("mipnerf360", root="data/", scene="garden")
dataset = load_dataset("tanks_and_temples", root="data/", scene="truck")
dataset = load_dataset("dtu", root="data/", scene="scan65")

for sample in dataset:
    image  = sample.image        # (H, W, 3) numpy array
    camera = sample.camera       # PinholeCamera
    c2w    = sample.c2w          # (4, 4) camera-to-world
```

Planned datasets: MipNeRF-360, Tanks & Temples, DTU, ScanNet, Replica

---

### `nerfprep.metrics` — Standardized evaluation *(coming soon)*

Reproducible metrics with citations. No more wondering which implementation a paper used.

```python
from nerfprep.metrics import evaluate

results = evaluate(renders, gt_images)
print(results)
# {
#   'psnr':  32.14,   # dB
#   'ssim':   0.921,  # Wang et al. 2004
#   'lpips':  0.087,  # Zhang et al. 2018, VGG backbone
# }
```

Each metric documents exactly which implementation it matches so your numbers are reproducible.

---

### `nerfprep.io` — Format conversion *(coming soon)*

Move between the formats researchers actually use.

```python
from nerfprep.io import load_ply, save_transforms_json, load_cameras_json

# Load a trained 3DGS scene
gaussians = load_ply("point_cloud.ply")

# Convert COLMAP poses to NeRFStudio / instant-ngp format
scene = load_colmap("sparse/0")
save_transforms_json(scene, "transforms.json")   # instant-ngp
save_cameras_json(scene, "cameras.json")          # nerfstudio
```

---

## Roadmap

### v0.1 — Foundation *(in progress)*
- [x] COLMAP binary + text parser (`cameras`, `images`, `points3D`)
- [x] Camera dataclasses with `K`, `R`, `c2w`, `w2c`, `camera_center`
- [x] Quaternion ↔ rotation matrix utilities
- [ ] Pinhole camera model (`project`, `unproject`)
- [ ] Coordinate convention converter (OpenCV ↔ OpenGL ↔ COLMAP ↔ Blender)
- [ ] PSNR, SSIM, LPIPS evaluation with standardized implementations

### v0.2 — Dataset loaders
- [ ] MipNeRF-360 dataset loader
- [ ] Tanks and Temples dataset loader
- [ ] DTU dataset loader
- [ ] ScanNet dataset loader

### v0.3 — Format I/O
- [ ] `transforms.json` read/write (instant-ngp)
- [ ] `cameras.json` read/write (nerfstudio)
- [ ] PLY point cloud read/write
- [ ] OBJ mesh basic support

### v0.4 — Developer experience
- [ ] CLI tools (`nerfprep inspect`, `nerfprep convert`)
- [ ] Visualization helpers (pose plots, point clouds)
- [ ] Comprehensive documentation site

### Future
- [ ] Mojo-accelerated backends for compute bottlenecks
- [ ] ScanNet++ support
- [ ] Replica support

---

## Contributing

Contributions are very welcome — especially from researchers who know which datasets and pain points matter most.

If you've written the same COLMAP parser or dataset loader for the third time, that's a sign it belongs here. Open an issue describing what you'd add.

```bash
git clone https://github.com/yourusername/nerfprep.git
cd nerfprep
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT — use it in your research, your papers, your pipelines. No strings attached.

---

## Citation

If `nerfprep` saves you time in a research project, a mention in your acknowledgements section is always appreciated but never required.

```bibtex
@software{nerfprep,
  author  = {Your Name},
  title   = {nerfprep: Data preparation utilities for neural rendering},
  year    = {2025},
  url     = {https://github.com/yourusername/nerfprep}
}
```

---

*Built by a 3D vision researcher, for 3D vision researchers.*
