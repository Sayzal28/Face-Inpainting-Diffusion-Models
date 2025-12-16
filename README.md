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
