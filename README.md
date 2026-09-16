# Image Inpainting Using Diffusion Models

This repository contains the implementation and experimental work for the MSc Artificial Intelligence dissertation **“Image Inpainting Using Diffusion Models”**, completed at the **University of Surrey**.

The project explores how **pre-trained unconditional diffusion models** can be adapted for **high-quality image inpainting**, with a focus on balancing **visual fidelity, inference speed, and computational cost**.

---

## 📌 Project Overview

Image inpainting is the task of filling missing or corrupted regions of an image such that the completed image is visually realistic and semantically consistent with its surrounding context. Traditional approaches, including diffusion-based PDE methods, patch-based methods, and GAN-based models, often struggle with either semantic coherence, training instability, or scalability.

This work demonstrates that **diffusion models** offer a robust and stable alternative. By adapting unconditional diffusion models through architectural and algorithmic conditioning, the project achieves high-quality inpainting results while significantly improving inference speed.

---

## 🧠 Key Contributions

* Adaptation of **unconditional diffusion models** for image inpainting
* Mask-aware architectural modification of a **U-Net-based diffusion model**
* **Continuous ground-truth noise injection** during sampling to preserve known regions
* Comparison of **DDPM vs DDIM** sampling strategies
* Evaluation of **multiple noise schedulers** (Linear, Cosine, Quadratic)
* Exploration of **parameter-efficient fine-tuning** using LoRA
* Extensive quantitative and qualitative evaluation

---

## 🏗️ Methodology Summary

### Base Models

* OpenAI Improved DDPM (ImageNet, 256×256)
* FFHQ pre-trained diffusion model (faces)

### Architectural Modifications

* U-Net input layer expanded to **9 channels**:

  * 3 × RGB image
  * 3 × binary mask
  * 3 × masked RGB image
* Timestep embeddings and self-attention retained

### Conditioning Strategy

* Inpainting is enforced **during inference**
* At each reverse diffusion step:

  * Known regions are re-injected with ground-truth noise
  * Prevents content drift and preserves original pixels

### Sampling Methods

* **DDPM** (1000 steps, baseline)
* **DDIM** (50–100 steps, accelerated inference)

### Noise Schedulers

* Linear
* Cosine (best-performing)
* Quadratic

### Fine-Tuning

* Full fine-tuning
* LoRA-based parameter-efficient fine-tuning applied to attention layers

---

# 🔁 Reproducing this on your machine

## 1. Prerequisites

| Requirement | Detail |
| :--- | :--- |
| Python | 3.10 or 3.12 (both were used during development) |
| GPU | A CUDA GPU with **≥ 16 GB VRAM** for training at 256×256. The code falls back to CPU, but a single DDPM sampling pass takes ~33 s per image on GPU and is impractical on CPU. |
| Disk | ~5 GB for checkpoints, plus whatever your image and mask datasets need |

## 2. Get the code

```bash
git clone https://github.com/Sayzal28/Face-Inpainting-Diffusion-Models.git
cd Face-Inpainting-Diffusion-Models/code
```

> **Run every script from inside `code/`.** The training and evaluation scripts import
> `from data.dataset import ...` and `from train_inpainting import ...` as top-level
> modules, which only resolve when `code/` is the working directory. The
> `sys.path.append(Path(__file__).parent.parent)` line at the top of those scripts points
> one level too high and does not help. There are also no `__init__.py` files in `data/`,
> `utils/` or `scripts/`, so importing the project as a package (`import code`) will fail —
> use the scripts as entry points.

## 3. Install the dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers training *and* evaluation:

```
torch>=1.9.0   torchvision>=0.10.0   torchaudio>=0.9.0   Pillow>=8.0.0
matplotlib>=3.5.0   opencv-python>=4.5.0   numpy>=1.21.0   scipy>=1.7.0   tqdm>=4.60.0
lpips>=0.1.4   pytorch-fid>=0.3.0   scikit-image>=0.18.0   torchmetrics>=0.7.0
```

(`requirements_1.txt` is a reduced, training-only subset — use `requirements.txt` unless
you have a reason not to.)

`lpips`, `pytorch-fid` and `scikit-image` are only needed for `--calculate_metrics`. If
`pytorch-fid` is missing the evaluation scripts still run and simply skip FID, printing a
warning.

## 4. Get the base checkpoint

Training starts from a **pre-trained unconditional 256×256 diffusion model** — the FFHQ
face model for the face experiments. This checkpoint is not included in the repository and
must be downloaded separately.

`create_model_and_diffusion()` in `train_inpainting.py` builds the UNet it will load the
weights into with exactly this architecture, so your checkpoint must match it:

```python
UNetModel(
    image_size=256,      in_channels=3,           model_channels=128,
    out_channels=6,      num_res_blocks=1,        attention_resolutions=(16,),
    channel_mult=(1, 1, 2, 2, 4, 4),              num_heads=4,
    num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True,
)
```

The checkpoint is loaded with `strict=False`, so a mismatched architecture will **not**
raise an error — it will silently load almost nothing and train from noise. Watch the line
it prints:

```
Missing keys: 0, Unexpected keys: 0
```

If those counts are large, your checkpoint does not match the architecture above and any
result you get will be meaningless.

The first `Conv2d` is then widened from 3 to **9 input channels** by
`DiffusionInpaintingModel` (RGB image + 3-channel mask + masked RGB), and the extra
channels are newly initialised.

## 5. Prepare the data

Two dataset trees are needed: **images** and **masks**.

```
data/
├── train/     ← flat directory of images (no class subfolders)
├── val/
└── test/

mask_dataset_split/
├── train/     ← flat directory of mask images
├── val/
└── test/
```

* Image directories are read **flat** — every `.jpg`, `.png`, `.jpeg`, `.bmp` or `.tiff`
  directly inside the directory, sorted by filename. Subfolders are ignored.
* The mask directory must contain a **subfolder per split**. `InpaintingDataset` raises
  `ValueError: Mask split directory not found` if `<mask_dir>/<split>` is missing.
* Masks are paired with images **serially**, not randomly: image *i* gets mask
  *i mod n_masks*, with the mask list repeating as needed. This makes the pairing identical
  across runs, so evaluation numbers are comparable between configurations.
* Everything is resized to `--img_size` (256) and normalised to [-1, 1].

The datasets used for the dissertation were **CelebA-HQ** (30,000 images) and a
**Places365** subset (36,500 images), split 75/15/10.

> **The mask generator is not in this repository.** The report describes procedurally
> generated masks covering 5%–60% of image area with uniqueness enforced, but no script
> that produces them is included — only the code that *reads* masks from disk. To reproduce
> the published numbers exactly you would need those exact mask files; to reproduce the
> method, generate your own binary masks (white = region to inpaint) into the layout above.

## 6. Train

```bash
python scripts/train.py \
  --checkpoint_path /path/to/ffhq_baseline.pt \
  --train_dir /path/to/data/train \
  --val_dir   /path/to/data/val \
  --mask_dir  /path/to/mask_dataset_split \
  --save_dir  checkpoints_enhanced \
  --batch_size 4 \
  --lr 5e-5 \
  --epochs 10 \
  --img_size 256
```

> **All four path arguments must be supplied.** Their defaults are absolute paths from the
> machine the work was done on (`/user/HS402/zs00774/Downloads/...`) and will not exist on
> yours. The same is true of every evaluation script below.

Useful additional flags:

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--scheduler_type` | `cosine` | `cosine`, `step` or `none` |
| `--warmup_epochs` | `0` | Warmup for the cosine schedule |
| `--min_lr_ratio` | `0.01` | Floor of the cosine schedule, as a ratio of the initial LR |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient-norm clipping |
| `--early_stopping_patience` | `5` | Epochs without improvement before stopping |
| `--max_checkpoints` | `2` | Numbered checkpoints kept on disk |
| `--resume` | — | A checkpoint path, or the literal `latest` / `best` |
| `--num_workers` | `4` | Dataloader workers |

**Outputs**, written to `--save_dir`:

```
latest_model.pt            overwritten every save
best_model.pt              lowest validation loss so far
checkpoint_epoch_<N>.pt    the last --max_checkpoints numbered checkpoints
```

Each file contains `epoch`, `model_state_dict`, `optimizer_state_dict`,
`scheduler_state_dict`, `val_loss` and a `diffusion_config` dict.

### Which training script to use

| Script | Noise schedule |
| :--- | :--- |
| `train_inpainting.py` (used by `scripts/train.py`) | **quadratic** |
| `train_inpainting_ddpm.py` | **linear** |

Both are libraries of functions with no `__main__` block — `scripts/train.py` is the only
training entry point. Note that the schedule is **hard-coded** inside
`create_model_and_diffusion()`, not exposed as a flag: to train with a different schedule,
edit the `noise_schedule=` argument at `train_inpainting.py:251`. The available names are
defined in `utils/schedules.py`.

## 7. Evaluate and sample

The evaluation scripts are variants of one another with different sampling defaults. They
share a common command-line interface:

```bash
python test_inpainting_better.py \
  --test_dir   /path/to/data/test \
  --mask_dir   /path/to/mask_dataset_split \
  --checkpoint_path   /path/to/ffhq_baseline.pt \
  --trained_checkpoint checkpoints_enhanced/best_model.pt \
  --output_dir results_ddim_100 \
  --use_ddim \
  --ddim_timesteps 100 \
  --calculate_metrics \
  --batch_size 4
```

Both checkpoints are required: `--checkpoint_path` rebuilds the base architecture, and
`--trained_checkpoint` supplies your fine-tuned inpainting weights.

| Script | Sampler | `--ddim_timesteps` default | Corresponds to |
| :--- | :--- | ---: | :--- |
| `tes_ddpm.py` | DDPM (pass no `--use_ddim`) | 100 | The DDPM row of the results table |
| `test_inpainting_better.py` | DDIM when `--use_ddim` is passed | 100 | The DDIM-100 row |
| `test_inp_ddim_50.py` | DDIM when `--use_ddim` is passed | 50 | The DDIM-50 row |
| `test_inp_ddim_100.py` | DDIM **on by default** | 50 | η = 0.75 variant |
| `test_ddim_30_cos.py` | DDIM when `--use_ddim` is passed | 30 | Fewest-steps ablation |
| `test_inpainting_better_n.py` | DDIM when `--use_ddim` is passed | — | Alternative mask directory |
| `test_quant.py` | DDIM + quantisation | — | Adds `--quantize`, `--use_half_precision`, `--compile_model`, `--fast_inference` |

Shared flags worth knowing:

| Flag | Purpose |
| :--- | :--- |
| `--use_injection` | Re-inject ground-truth noise into known regions at each reverse step |
| `--ddim_eta` | DDIM stochasticity; `0.0` is deterministic |
| `--blend_output` | Composite the generated region back onto the original image |
| `--calculate_metrics` | Compute FID, LPIPS and SSIM |
| `--quick_test` / `--max_batches N` | Run on a small subset first |
| `--random_samples` / `--seed` | Sample selection and reproducibility |
| `--clip_denoised` | Clamp predictions to the valid range |

**Start with `--quick_test`** to confirm your paths and checkpoints are right before
launching a full evaluation run — a complete DDPM pass over a test set is slow.

> Filenames such as `test_ddim_30_cos.py` suggest a cosine schedule, but every evaluation
> script builds its diffusion through `create_model_and_diffusion()` and therefore inherits
> the **quadratic** schedule set there. Change it at the source if you need a different one.

## 8. Visualising the noise schedules

```bash
python noise.py --image_path /path/to/an/image.png --output_dir noise_schedule_visualization
```

Writes `noise_schedule_comparison.png`, a grid comparing the schedules across timesteps.

## 9. What you should get

With the FFHQ base model and the DDIM-100 configuration, the numbers to aim for are the
ones in the results table below — FID ≈ 3.24, LPIPS ≈ 0.047, SSIM ≈ 0.921 at ≈ 3.42 s per
sample. Timings are hardware-dependent; the FID/LPIPS/SSIM values are not, provided the
image and mask sets match.

---

## 📊 Results Overview

| Method | Steps | FID ↓    | LPIPS ↓ | SSIM ↑    | Time / Sample |
| ------ | ----- | -------- | ------- | --------- | ------------- |
| DDPM   | 1000  | 3.70     | 0.046   | 0.906     | 33.41s        |
| DDIM   | 100   | **3.24** | 0.047   | **0.921** | **3.42s**     |
| DDIM   | 50    | 3.62     | 0.047   | 0.910     | 1.75s         |

**Key findings:**

* DDIM with **100 steps + Cosine scheduler** provides the best quality–speed trade-off
* Domain-specific pre-training (FFHQ) significantly improves face inpainting
* Sampling time reduced by **~10×** compared to DDPM

---

## 📂 Datasets

* **CelebA-HQ** (30,000 images)
* **Places365** (36,500-image subset)

Datasets were split into **train / validation / test (75/15/10)**.
Procedurally generated masks covering **5%–60%** of image area were used, with strict uniqueness enforcement.

> ⚠️ Datasets are not included in this repository and must be obtained separately.

---

## 📈 Evaluation Metrics

* **FID (Frechet Inception Distance)** – realism and diversity
* **LPIPS** – perceptual similarity
* **SSIM** – structural consistency
* Qualitative visual inspection

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* PEFT (LoRA)
* LPIPS
* pytorch-fid
* scikit-image

---

## 🔖 Code Attribution and Acknowledgements

This project **partially builds upon the structural design and foundational components** of OpenAI’s implementation of *Improved Denoising Diffusion Probabilistic Models (DDPMs)*.

* The **overall project structure and baseline diffusion framework** were initially inspired by the official OpenAI codebase accompanying:

  > *Nichol & Dhariwal, “Improved Denoising Diffusion Probabilistic Models” (ICML 2021)*

* Beyond this structural reference, **the majority of the codebase was modified, extended, or newly implemented** as part of this research, including:

  * Mask-aware architectural changes (9-channel U-Net input)
  * Custom data pipelines and procedural mask generation
  * Inpainting-specific conditioning via continuous ground-truth noise injection
  * Integration and evaluation of DDIM sampling
  * Multiple noise schedulers and ablation studies
  * Parameter-efficient fine-tuning using LoRA
  * Custom training, validation, and evaluation workflows

The OpenAI implementation served as a **starting structural reference**, while **all inpainting logic, experimental methodology, and optimizations were developed independently** for this project.
This acknowledgement is provided for **academic transparency and proper attribution**.

---

## 🎓 Academic Context

* **Degree**: MSc Artificial Intelligence
* **Institution**: University of Surrey
* **Supervisor**: Dr. Marco Volino
* **Author**: Saizalpreet Kaur
* **Year**: 2025

---

## 📌 Citation

If you reference or build upon this work, please cite:

> Saizalpreet Kaur, *Image Inpainting Using Diffusion Models*, MSc Dissertation, University of Surrey, 2025.

---

## ⚠️ Notes

* This repository is intended for **academic and research purposes**
* Trained model weights may be omitted due to size constraints
* Some experiments require high-memory GPUs

---

## 📬 Contact

For questions, issues, or collaboration, please open an issue on GitHub.
