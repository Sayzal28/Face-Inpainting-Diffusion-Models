# diffusion_inpainting/__init__.py
"""
Diffusion Inpainting Package

A PyTorch implementation of diffusion models for image inpainting.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .unet import UNetModel, DiffusionInpaintingModel
from .gaussian_diffusion import GaussianDiffusion
from .losses import ModelMeanType, ModelVarType, LossType
from .data.dataset import InpaintingDataset, create_inpainting_dataloaders
from .utils.schedules import get_named_beta_schedule, create_gaussian_diffusion

__all__ = [
    "UNetModel",
    "DiffusionInpaintingModel", 
    "GaussianDiffusion",
    "ModelMeanType",
    "ModelVarType", 
    "LossType",
    "InpaintingDataset",
    "create_inpainting_dataloaders",
    "get_named_beta_schedule",
    "create_gaussian_diffusion"
]

# data/__init__.py
from .dataset import InpaintingDataset, InpaintingDatasetWithClassConditioning, create_inpainting_dataloaders

__all__ = ["InpaintingDataset", "InpaintingDatasetWithClassConditioning", "create_inpainting_dataloaders"]

# utils/__init__.py
from .schedules import get_named_beta_schedule, betas_for_alpha_bar, create_gaussian_diffusion

__all__ = ["get_named_beta_schedule", "betas_for_alpha_bar", "create_gaussian_diffusion"]

# scripts/__init__.py
"""Training scripts for diffusion inpainting models."""

__all__ = []
