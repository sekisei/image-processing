# image-processing / SBSM

[日本語README](README.md)

This project runs SBSM (Statistical Background Subtraction Model) on CUDA.
The implementation lives under the SBSM folder.

## Demo Result

Input image used:
- readme-images/sample.png

Output image generated:
- readme-images/sbsm_gpu.jpg

### Input

![Input sample](readme-images/sample.png)

### Output

![Output sbsm_gpu](readme-images/sbsm_gpu.jpg)

### What to look at

- Red overlay indicates foreground-like pixels detected by SBSM.
- Background regions mostly keep the original appearance.
- Even in noisy night scenes, motion/change regions are emphasized.

## Key Files

- Main script: SBSM/src/cuda_sbsm.py
- Input images: SBSM/data/input/
- Output images: SBSM/data/output/
- Background histogram: SBSM/data/hist_xy.npy

## Run (Podman Compose)

1. Move into SBSM directory

```bash
cd SBSM
```

2. Build dev container

```bash
podman-compose -f docker-compose.dev.yml build
```

3. Run interactively

```bash
podman-compose -f docker-compose.dev.yml run --rm sbsm bash
python3 src/cuda_sbsm.py
```

For one-shot production-style run:

```bash
podman-compose -f docker-compose.prod.yml up --build
```

## GPU Check

Run this under SBSM:

```bash
podman-compose -f docker-compose.dev.yml run --rm sbsm bash -lc "nvidia-smi -L && python3 -c 'from numba import cuda; print(cuda.is_available())'"
```

Expected:
- GPU name is listed by nvidia-smi -L
- cuda.is_available() returns True

## Notes

- cuda_sbsm.py currently loads SBSM/data/hist_xy.npy by default.
- Histogram generation from background frames is present in code but currently commented out.
