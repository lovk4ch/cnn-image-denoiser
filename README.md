# CNN-based Image Denoiser
*High-Performance Local-Context Neural Image Restoration Model*

<div align="center">
  <a href="https://cnn-image-denoiser.onrender.com">
    <b>Live demo is available here</a>
    <br>
  ❗️ Note: try to use images lower than 512x512 to prevent memory overrun
    </b>
</div>

---

## Project Overview

**CNN-based Image Denoiser** is a neural network model for image denoising based on **convolutional neural networks (CNNs)**.
The model efficiently restores images using **local patterns and textures**, making it ideal for **real-time and embedded systems** where speed and stability are critical.

The project demonstrates a classic **low-level computer vision** approach, comparable to traditional filters and modern transformers.

⚠️ This implementation is a demonstration of convolutional neural networks and does not currently represent a universal solution for defect removal. For different types of photos with artifacts of varying structure a good result is not guaranteed.

---

## Processing examples

**The example photos were processed in two passes. Network parameters:**
* feature channels = 64
* kernel size = 3
* layers = 5
* dilation = [1, 1, 1, 2, 1, 1, 1]
* image size = 1024

<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_003_layers%3D3_channels%3D48.jpg" width="500"><br>
  <em>Noise between bright and dark areas</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_001_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Local cleaning of noisy areas</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_002_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>ISO noise and grain reduction</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_004_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Removing defects and restoring the picture</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_005_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Smoothing details from point-and-shoot cameras</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_006_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Removing fine roughness</em>
</p>

---

## Objective

* Additive noise removal (Gaussian/synthetic noise)
* Local detail restoration

Comparison with:
* Classic filters (Gaussian, Median)
* Baseline (Blur Filters)

---

## Model Architecture

```
Input Image
   ↓
Convolution Block × N
   ↓
Residual Connections
   ↓
BatchNorm + ReLU
   ↓
Final Convolution
   ↓
Denoised Image
```

---

## Key Components

* **3x3 / 5x5 convolution layers**
* **Residual blocks** for a stable gradients
* **ReLU / LeakyReLU activations**
* **Batch Normalization**

---

## Advantages over Blur Filters

### ❌ Classical Blur Filters

* Gaussian / Median / Bilateral
* Works only locally
* Blurs the boundaries
* Doesn't adapt to the scene structure

### 🚀 Key Advantages

* **Preserves local structures and textures**
* **Adaptable to different noise levels**
* **Fast inference suitable for embedded systems**

### ✅ CNN-based Denoiser

| Feature                       | Blur Filter | CNN |
| ----------------------------- | ----------- | --- |
| Local pattern recognition     | ⚡           | ✅   |
| Edge / structure preservation | ❌           | ✅   |
| Adaptive learning             | ❌           | ✅   |
| Real-time inference           | ✅           | ⚡   |
| Training on dataset           | ❌           | ✅   |

### ⚠️ Trade-offs
* Limited global context (patch-to-patch long-range)
* Doesn't always perfectly reconstruct complex noise patterns

---

## 📊 Evaluation Metrics

### Image Quality

* **PSNR** — Peak Signal-to-Noise Ratio
* **SSIM** — Structural Similarity Index

---

## Dataset

* Synthetic noisy DSLR-images

## Noise Model
* Gaussian Noise (σ configurable)
* Luminance ISO noise
* Chroma noise
* Various mixes of all types

---

## Tech Stack

* 🐍 Python
* 🔥 PyTorch
* 🖥️ Custom CNN architecture
* 📊 TensorBoard

---

## Key Takeaways

* ✔️ Demonstrates **local-pattern-based denoising**
* ✔️ Shows **practical use of CNNs in applied CV**
* ✔️ Provides **baseline for comparison with transformer-based denoisers**
* ✔️ Relevant for **Applied ML / Computer Vision roles**

---

## 🔀 Next Releases:

* Add attention layers for extended context (CNN + Attention)
* Move towards diffusion-based denoising
* Add FiLM filters to adjust the level and type of noise
* Add transformed-based restoration models net for the next improvement of out images

## Installation

Use Python 3.11 or 3.12 and a separate virtual environment. The base requirements file contains direct project dependencies; choose one PyTorch profile for Windows/Linux x86-64.

NVIDIA RTX 2070 (CUDA 12.1 build; requires a compatible NVIDIA driver):

```powershell
python -m pip install -r requirements-cuda.txt
```

CPU:

```powershell
python -m pip install -r requirements-cpu.txt
```

Verify the selected environment:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip check
```

The profiles pin torch 2.5.1 and torchvision 0.20.1. For GPUs requiring newer builds, choose a supported matched pair and wheel index from the [official PyTorch installer](https://pytorch.org/get-started/locally/). CUDA builds depend on GPU architecture and driver compatibility; a separate wheel is not needed for every card model. Installing a CUDA toolkit alone does not convert CPU wheels into CUDA wheels. The Dockerfile uses the CPU profile.

`requirements.txt` contains direct dependencies, not a complete lock file. Transitive dependencies are resolved by pip. Evaluation uses data/test as a validation set: fixed Gaussian noise (sigma 0.03), a deterministic center crop up to 512 pixels, and a single denoising pass. Keep these photos and their scenes independent from training; reserve another untouched set for final testing.

## Train and evaluation

```powershell
python src/main.py --mode train
python src/main.py --mode eval
```

`eval.enabled: true` runs evaluation after each training epoch. Set it to `false` to train without evaluation. Standalone `--mode eval` always runs evaluation using exported `denoise.pth` weights. Original test images are not modified. Reports contain train and eval PSNR; evaluation image reports also include noisy input PSNR. Evaluation samples are saved under `eval_epoch_#####` (or `eval` for standalone runs).

Checkpoints: `denoise_cpt.pth` resumes the latest training state; `denoise_cpt_best_train.pth` and `denoise_cpt_best_eval.pth` store full states selected independently by maximum train/eval PSNR. `denoise.pth` exports best eval weights for the API (best train weights when evaluation is disabled). Historical `denoise_cpt_best.pth` is no longer updated. Evaluation scores reset when the test files or evaluation settings change. Train PSNR is measured on stochastic training batches, so it is less suitable for model selection than fixed evaluation PSNR.

Photo import inventory: `data/train_import.csv` records source paths and pixel dimensions for copied training JPGs. Every training photo contributes `patches_per_image` random crops per epoch.