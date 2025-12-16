#!/usr/bin/env python3
"""
Inpainting sampler with noise injection during sampling.
Uses dataset utilities from data.dataset module.
Now includes speed optimizations and quantization support.
"""

import os
import sys
import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import lpips
from skimage.metrics import structural_similarity as ssim

# Quantization imports
import torch.quantization as quantization
try:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: Advanced quantization not available. Only simple quantization will work.")

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_inference_dataloader, OrderedMaskDataset, FlatImageDataset
from train_inpainting import create_model_and_diffusion  # Standard model loading

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-fid not installed. FID calculation will be skipped.")
    print("Install with: pip install pytorch-fid")
    FID_AVAILABLE = False


def toU8(sample):
    """Convert tensor to uint8 numpy array."""
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def save_results(srs, gts, masked_imgs, masks, img_names, output_dir, batch_idx):
    """Save sampling results with comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = len(img_names) if img_names else srs.shape[0]
    
    for i in range(batch_size):
        # Individual results
        if img_names:
            base_name = Path(img_names[i]).stem
        else:
            base_name = f"batch_{batch_idx}_sample_{i}"
        
        # Save individual images
        if srs is not None:
            save_path = output_dir / f"{base_name}_inpainted.png"
            Image.fromarray(srs[i]).save(save_path)
        
        if gts is not None:
            save_path = output_dir / f"{base_name}_original.png"
            Image.fromarray(gts[i]).save(save_path)
        
        if masked_imgs is not None:
            save_path = output_dir / f"{base_name}_masked.png"
            Image.fromarray(masked_imgs[i]).save(save_path)
    
    # Save comparison grid - ONLY 3 ROWS (original, masked, inpainted)
    comparison_list = []
    if gts is not None:
        comparison_list.append(torch.from_numpy(gts).permute(0, 3, 1, 2) / 127.5 - 1)
    if masked_imgs is not None:
        comparison_list.append(torch.from_numpy(masked_imgs).permute(0, 3, 1, 2) / 127.5 - 1)
    if srs is not None:
        comparison_list.append(torch.from_numpy(srs).permute(0, 3, 1, 2) / 127.5 - 1)
    
    if comparison_list:
        comparison = torch.cat(comparison_list, dim=0)
        save_image(
            comparison,
            output_dir / f"comparison_batch_{batch_idx}.png",
            nrow=batch_size,
            normalize=True,
            value_range=(-1, 1)
        )


class SimpleQuantizedModel:
    """Simple wrapper for INT8 quantization."""
    
    def __init__(self, model, device):
        self.device = device
        self.original_model = model
        self.quantized_model = None
        self._quantize_simple()
    
    def _quantize_simple(self):
        """Apply simple dynamic quantization."""
        print("Applying simple INT8 quantization...")
        
        # Move to CPU for quantization (required for most quantization methods)
        cpu_model = self.original_model.cpu()
        
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            cpu_model,
            {torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.GroupNorm},
            dtype=torch.qint8
        )
        
        # Keep on CPU - most quantized models work better on CPU
        print("Quantized model will run on CPU for optimal performance")
        
        print("Simple quantization applied successfully")
    
    def __call__(self, *args, **kwargs):
        # Move inputs to CPU for quantized model
        args_cpu = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_cpu.append(arg.cpu())
            else:
                args_cpu.append(arg)
        
        kwargs_cpu = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs_cpu[k] = v.cpu()
            else:
                kwargs_cpu[k] = v
        
        result = self.quantized_model(*args_cpu, **kwargs_cpu)
        
        # Move result back to original device
        if isinstance(result, torch.Tensor):
            return result.to(self.device)
        return result


def create_ddim_timesteps(num_ddim_steps, num_train_timesteps=1000):
    """Create custom timestep schedule for DDIM."""
    step_ratio = num_train_timesteps // num_ddim_steps
    timesteps = (np.arange(0, num_ddim_steps) * step_ratio).round().astype(int)
    timesteps = timesteps[::-1]  # Reverse order
    return timesteps.tolist()


class MetricsCalculator:
    """Calculate various image quality metrics."""
    
    def __init__(self, device):
        self.device = device
        
        # Initialize LPIPS only once
        print("Initializing LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()  # Set to eval mode
        print("LPIPS model loaded successfully")
        
        # Storage for all images
        self.all_original = []
        self.all_inpainted = []
    
    def add_batch(self, original_images, inpainted_images):
        """Add a batch of images for metric calculation."""
        # Convert to [0, 1] range and store
        orig_01 = (original_images + 1) / 2  # [-1, 1] to [0, 1]
        inpaint_01 = (inpainted_images + 1) / 2
        
        self.all_original.append(orig_01.cpu())
        self.all_inpainted.append(inpaint_01.cpu())
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS between two images."""
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            lpips_dist = self.lpips_fn(img1, img2)
            return lpips_dist.mean().item()
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images."""
        # Convert to numpy and [0, 1] range
        img1_np = ((img1.cpu() + 1) / 2).numpy()
        img2_np = ((img2.cpu() + 1) / 2).numpy()
        
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            # Convert from CHW to HWC for SSIM
            img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
            img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
            
            # Calculate SSIM
            score = ssim(img1_hwc, img2_hwc, multichannel=True, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    def save_images_for_fid(self, output_dir):
        """Save images in format required for FID calculation."""
        if not FID_AVAILABLE:
            return None, None
            
        # Create directories for FID
        real_dir = Path(output_dir) / "fid_real"
        fake_dir = Path(output_dir) / "fid_generated"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all images
        all_real = torch.cat(self.all_original, dim=0)
        all_fake = torch.cat(self.all_inpainted, dim=0)
        
        print(f"Saving {len(all_real)} images for FID calculation...")
        
        # Save images
        for i, (real_img, fake_img) in enumerate(zip(all_real, all_fake)):
            # Convert to PIL format [0, 255]
            real_pil = transforms.ToPILImage()(real_img)
            fake_pil = transforms.ToPILImage()(fake_img)
            
            real_pil.save(real_dir / f"real_{i:04d}.png")
            fake_pil.save(fake_dir / f"fake_{i:04d}.png")
        
        return str(real_dir), str(fake_dir)
    
    def calculate_fid(self, real_dir, fake_dir):
        """Calculate FID score."""
        if not FID_AVAILABLE:
            return None
            
        try:
            print("Calculating FID score...")
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=50,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return None
    
    def calculate_all_metrics(self, output_dir):
        """Calculate all metrics and return results."""
        if not self.all_original or not self.all_inpainted:
            print("No images available for metric calculation")
            return {}
        
        print(f"\nCalculating metrics for {len(torch.cat(self.all_original, dim=0))} images...")
        results = {}
        
        # Concatenate all images
        all_real = torch.cat(self.all_original, dim=0).to(self.device)
        all_fake = torch.cat(self.all_inpainted, dim=0).to(self.device)
        
        # Convert back to [-1, 1] for LPIPS
        all_real_lpips = all_real * 2 - 1
        all_fake_lpips = all_fake * 2 - 1
        
        # Calculate LPIPS
        print("Calculating LPIPS...")
        lpips_scores = []
        batch_size = 8  # Smaller batch to avoid memory issues
        for i in range(0, len(all_real), batch_size):
            end_idx = min(i + batch_size, len(all_real))
            batch_real = all_real_lpips[i:end_idx]
            batch_fake = all_fake_lpips[i:end_idx]
            
            lpips_batch = self.calculate_lpips(batch_real, batch_fake)
            lpips_scores.append(lpips_batch)
        
        results['LPIPS'] = np.mean(lpips_scores)
        
        # Calculate SSIM
        print("Calculating SSIM...")
        ssim_scores = []
        for i in range(0, len(all_real), batch_size):
            end_idx = min(i + batch_size, len(all_real))
            batch_real = all_real_lpips[i:end_idx]  # Use lpips version ([-1,1])
            batch_fake = all_fake_lpips[i:end_idx]
            
            ssim_batch = self.calculate_ssim(batch_real, batch_fake)
            ssim_scores.append(ssim_batch)
        
        results['SSIM'] = np.mean(ssim_scores)
        
        # Calculate FID
        if FID_AVAILABLE:
            print("Preparing images for FID...")
            real_dir, fake_dir = self.save_images_for_fid(output_dir)
            if real_dir and fake_dir:
                fid_value = self.calculate_fid(real_dir, fake_dir)
                if fid_value is not None:
                    results['FID'] = fid_value
        
        return results


class InpaintingSampler:
    """Professional inpainting sampler with noise injection and optimizations."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
            print(f"Set random seed to {args.seed}")
        
        # Load model
        self._load_model()
        
        # Create dataloader
        self._create_dataloader()
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics calculator
        if args.calculate_metrics:
            self.metrics_calc = MetricsCalculator(self.device)
            print("Metrics calculator initialized")
        else:
            self.metrics_calc = None
    
    def _load_model(self):
        """Load standard model with optimizations."""
        print("Loading standard model...")
        self.model, self.diffusion, _ = create_model_and_diffusion(
            checkpoint_path=self.args.checkpoint_path,
            device=self.device,
            img_size=self.args.img_size
        )
        
        # Load trained checkpoint if provided
        if self.args.trained_checkpoint:
            print(f"Loading trained weights from {self.args.trained_checkpoint}")
            
            checkpoint = torch.load(self.args.trained_checkpoint, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        
        # Apply optimizations
        self._apply_optimizations()
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _apply_optimizations(self):
        """Apply various optimizations to the model."""
        # Half precision
        if self.args.use_half_precision and self.device.type == 'cuda':
            print("Applying half precision (FP16)...")
            self.model = self.model.half()
        
        # Model compilation (PyTorch 2.0+)
        if self.args.compile_model and hasattr(torch, 'compile'):
            print("Compiling model for faster inference...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # Quantization
        if self.args.quantize != 'none':
            print(f"Applying {self.args.quantize} quantization...")
            if self.args.quantize == 'simple':
                self.model = SimpleQuantizedModel(self.model, self.device)
            else:
                print("Warning: Advanced quantization not fully implemented. Using simple quantization.")
                self.model = SimpleQuantizedModel(self.model, self.device)
    
    def _create_dataloader(self):
        """Create dataloader for inpainting with ordered mask cycling."""
        print("Creating dataloader with ordered mask cycling...")
        
        # Use the convenient function from dataset.py
        self.dataloader = create_inference_dataloader(
            test_dir=self.args.test_dir,
            mask_dir=self.args.mask_dir,
            batch_size=self.args.batch_size,
            img_size=self.args.img_size,
            num_workers=self.args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            random_samples=self.args.random_samples
        )
        
        print(f"Total batches: {len(self.dataloader)}")
    
    def model_fn(self, x, t, gt=None, gt_keep_mask=None, **kwargs):
        """Model function for inpainting conditioning."""
        if gt is None or gt_keep_mask is None:
            raise ValueError("Ground truth and mask required for inpainting")
        
        # Create masked image (known regions)
        masked_image = gt * gt_keep_mask + torch.zeros_like(gt) * (1 - gt_keep_mask)
        
        # Create inpainting mask (1 = inpaint, 0 = keep)
        inpaint_mask = 1 - gt_keep_mask
        
        # Call your inpainting model
        return self.model(x, t, masked_image=masked_image, mask=inpaint_mask)
    
    def inpainting_ddim_sample_loop_fast(self, model_fn, shape, gt_images, masks,
                                        clip_denoised=True, device=None, progress=False, 
                                        eta=0.0, num_steps=None):
        """Fast DDIM sampling with configurable steps."""
        if device is None:
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.device
        
        assert isinstance(shape, (tuple, list))
        
        # Use fewer steps if specified
        total_steps = num_steps or getattr(self.args, 'ddim_steps', 50)
        
        # Create custom timestep schedule
        if total_steps >= self.diffusion.num_timesteps:
            indices = list(range(self.diffusion.num_timesteps))[::-1]
        else:
            indices = create_ddim_timesteps(total_steps, self.diffusion.num_timesteps)
        
        if progress:
            indices = tqdm(indices, desc=f"Fast DDIM ({len(indices)} steps)")
        
        # Start with pure noise
        img = torch.randn(*shape, device=device)
        gt_keep_mask = 1 - masks
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            
            with torch.no_grad():
                # DDIM step
                model_kwargs = {
                    "gt": gt_images,
                    "gt_keep_mask": gt_keep_mask,
                }
                
                # Get model prediction
                out = self.diffusion.p_mean_variance(
                    model_fn, img, t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs
                )
                
                # DDIM sampling formula
                eps = self.diffusion._predict_eps_from_xstart(img, t, out["pred_xstart"])
                
                alpha_bar = self.diffusion.alphas_cumprod[i]
                alpha_bar_prev = self.diffusion.alphas_cumprod_prev[i]
                
                alpha_bar = torch.tensor(alpha_bar, device=device)
                alpha_bar_prev = torch.tensor(alpha_bar_prev, device=device)
                
                sigma = (
                    eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * 
                    torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                
                noise = torch.randn_like(img)
                mean_pred = (
                    out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + 
                    torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                )
                
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                img = mean_pred + nonzero_mask * sigma * noise
                
                # INPAINTING INJECTION (if enabled)
                if self.args.use_injection and i > 0:
                    alpha_cumprod_t = alpha_bar_prev
                    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
                    
                    known_noise = torch.randn_like(gt_images)
                    noised_known_regions = (
                        sqrt_alpha_cumprod * gt_images + 
                        sqrt_one_minus_alpha_cumprod * known_noise
                    )
                    
                    # Inject known regions
                    img = img * masks + noised_known_regions * gt_keep_mask
        
        return img
    
    def inpainting_p_sample_loop(self, model_fn, shape, gt_images, masks, 
                                 clip_denoised=True, device=None, progress=False):
        """DDPM sampling with inpainting noise injection."""
        if device is None:
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.device
        
        assert isinstance(shape, (tuple, list))
        
        # Start with pure noise
        img = torch.randn(*shape, device=device)
        
        # Get timestep indices in reverse order
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            indices = tqdm(indices, desc="DDPM inpainting with injection")
        
        # Prepare masks and known regions
        gt_keep_mask = 1 - masks  # Keep mask (1 = keep, 0 = inpaint)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            
            with torch.no_grad():
                # STEP 1: Regular diffusion step
                model_kwargs = {
                    "gt": gt_images,
                    "gt_keep_mask": gt_keep_mask,
                }
                
                # Get model prediction
                out = self.diffusion.p_mean_variance(
                    model_fn, img, t, 
                    clip_denoised=clip_denoised, 
                    model_kwargs=model_kwargs
                )
                
                # Sample from the predicted distribution
                noise = torch.randn_like(img)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                
                # Standard sampling step
                img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                
                # STEP 2: INPAINTING INJECTION
                if self.args.use_injection and i > 0:
                    # Calculate what the known regions should look like at this noise level
                    alpha_cumprod_t = self.diffusion.alphas_cumprod[i-1] if i > 0 else 1.0
                    alpha_cumprod_t = torch.tensor(alpha_cumprod_t, device=device)
                    
                    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
                    
                    # Add the right amount of noise to the original image
                    known_noise = torch.randn_like(gt_images)
                    noised_known_regions = (
                        sqrt_alpha_cumprod * gt_images + 
                        sqrt_one_minus_alpha_cumprod * known_noise
                    )
                    
                    # Inject: keep generated content in holes, use noised original in known regions
                    img = img * masks + noised_known_regions * gt_keep_mask
        
        return img

    @torch.inference_mode()  # Faster than torch.no_grad()
    def sample_batch(self, batch):
        """Sample inpainting for a batch with optimizations."""
        # Move batch to device
        gt_images = batch['image'].to(self.device)
        masked_images = batch['masked_image'].to(self.device) 
        masks = batch['mask'].to(self.device)
        
        # Apply half precision if enabled
        if self.args.use_half_precision and self.device.type == 'cuda':
            gt_images = gt_images.half()
            masked_images = masked_images.half()
            masks = masks.half()
        
        batch_size = gt_images.shape[0]
        
        # Print mask info for first batch
        if not hasattr(self, '_first_batch_printed'):
            self._first_batch_printed = True
            if 'mask_path' in batch:
                print(f"Using masks: {[Path(p).name for p in batch['mask_path']]}")
            if 'mask_idx' in batch:
                print(f"Mask indices: {batch['mask_idx'].tolist()}")
        
        if not self.args.fast_inference:
            print(f"Sampling batch of {batch_size} images...")
            print(f"GT images shape: {gt_images.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Device: {self.device}")
        
        # Choose sampling method
        if self.args.use_ddim or hasattr(self.args, 'ddim_steps'):
            sampling_method = f"Fast DDIM ({getattr(self.args, 'ddim_steps', 50)} steps)"
            if not self.args.fast_inference:
                print(f"Using {sampling_method}...")
            
            result = self.inpainting_ddim_sample_loop_fast(
                self.model_fn,
                (batch_size, 3, self.args.img_size, self.args.img_size),
                gt_images,
                masks,
                clip_denoised=self.args.clip_denoised,
                device=self.device,
                progress=self.args.show_progress and not self.args.fast_inference,
                eta=self.args.ddim_eta,
                num_steps=getattr(self.args, 'ddim_steps', None)
            )
        else:
            sampling_method = "DDPM with injection"
            if not self.args.fast_inference:
                print(f"Using {sampling_method}...")
            
            result = self.inpainting_p_sample_loop(
                self.model_fn,
                (batch_size, 3, self.args.img_size, self.args.img_size),
                gt_images,
                masks,
                clip_denoised=self.args.clip_denoised,
                device=self.device,
                progress=self.args.show_progress and not self.args.fast_inference
            )
        
        if not self.args.fast_inference:
            print("Sampling completed!")
        
        # Final blending
        if self.args.blend_output and not self.args.skip_final_blend:
            keep_mask = 1 - masks
            result = result * masks + gt_images * keep_mask
        
        # Clear cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result, gt_images, masked_images, masks
    
    def run_sampling(self):
        """Run complete sampling process."""
        injection_text = " with injection" if self.args.use_injection else " (original)"
        method_text = f"{'Fast DDIM' if self.args.use_ddim or hasattr(self.args, 'ddim_steps') else 'DDPM'}"
        quantize_text = f", {self.args.quantize} quantized" if self.args.quantize != 'none' else ""
        
        print(f"Starting inpainting sampling{injection_text}...")
        print(f"Sampling method: {method_text}{injection_text}{quantize_text}")
        print(f"Output directory: {self.output_dir}")
        
        total_time = 0
        total_samples = 0
        
        max_batches = self.args.max_batches if self.args.max_batches > 0 else len(self.dataloader)
        
        progress_bar = tqdm(self.dataloader, desc="Processing batches") if not self.args.fast_inference else self.dataloader
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= max_batches:
                break
            
            start_time = time.time()
            
            # Sample batch
            sampled, gt_images, masked_images, masks = self.sample_batch(batch)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += gt_images.shape[0]
            
            # Add to metrics calculator
            if self.args.calculate_metrics and self.metrics_calc:
                self.metrics_calc.add_batch(gt_images, sampled)
            
            # Convert to uint8 for saving
            srs = toU8(sampled)
            gts = toU8(gt_images)
            masked_imgs = toU8(masked_images)
            mask_vis = toU8(masks.repeat(1, 3, 1, 1))
            
            # Save results
            img_names = batch.get('image_path', None)
            save_results(
                srs=srs,
                gts=gts, 
                masked_imgs=masked_imgs,
                masks=mask_vis,
                img_names=img_names,
                output_dir=self.output_dir,
                batch_idx=batch_idx
            )
            
            if not self.args.fast_inference:
                print(f"Batch {batch_idx + 1}/{max_batches} completed in {batch_time:.2f}s")
        
        avg_time = total_time / total_samples if total_samples > 0 else 0
        print(f"\nSampling completed!")
        print(f"Total samples: {total_samples}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per sample: {avg_time:.2f}s")
        print(f"Results saved to: {self.output_dir}")
        
        # Calculate and save metrics
        if self.args.calculate_metrics and self.metrics_calc and total_samples > 0:
            print(f"\nCalculating metrics for {total_samples} samples...")
            metrics = self.metrics_calc.calculate_all_metrics(self.output_dir)
            
            # Print results
            print("\n" + "="*50)
            print("EVALUATION METRICS")
            print("="*50)
            for metric_name, value in metrics.items():
                print(f"{metric_name:10s}: {value:.4f}")
            print("="*50)
            
            # Save metrics to file
            metrics_file = self.output_dir / "metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write(f"Evaluation Metrics\n")
                f.write(f"==================\n")
                f.write(f"Total samples: {total_samples}\n")
                f.write(f"Total time: {total_time:.2f}s\n")
                f.write(f"Sampling method: {method_text}{injection_text}{quantize_text}\n")
                f.write(f"Avg time per sample: {avg_time:.2f}s\n\n")
                
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            print(f"Metrics saved to: {metrics_file}")
        else:
            print("Metrics calculation skipped (use --calculate_metrics to enable)")


def apply_speed_optimizations(args):
    """Apply speed optimization presets including quantization."""
    if args.fast_inference:
        print("Applying fast inference optimizations...")
        args.use_ddim = True
        args.ddim_steps = getattr(args, 'ddim_steps', 20)  # Use specified or default to 20
        args.batch_size = min(args.batch_size * 2, 16)  # Double batch size, cap at 16
        args.show_progress = False
        args.calculate_metrics = False
        args.skip_final_blend = True
        args.blend_output = False
        
        # Add quantization for maximum speed
        if args.quantize == 'none':
            args.quantize = 'simple'  # Auto-enable simple quantization
            print("Auto-enabled simple quantization for fast inference")
        
        print(f"Fast mode: {args.ddim_steps} DDIM steps, batch size {args.batch_size}, quantization: {args.quantize}")
    
    return args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inpainting sampler with optional noise injection, speed optimizations and quantization')
    
    # Data arguments
    parser.add_argument('--test_dir', type=str, default='/user/HS402/zs00774/Downloads/data/test',
                       help='Directory containing test images')
    parser.add_argument('--mask_dir', type=str, default='/user/HS402/zs00774/Downloads/masking',
                       help='Directory containing mask images')
    parser.add_argument('--train_dir', type=str, default='/user/HS402/zs00774/Downloads/data/train',
                       help='Train directory (dummy, required by dataloader)')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, default='/user/HS402/zs00774/Downloads/ffhq_baseline.pt',
                       help='Path to base model checkpoint')
    parser.add_argument('--trained_checkpoint', type=str, default='/user/HS402/zs00774/Downloads/idkyet_ffid/checkpoints_enhanced/latest_model.pt',
                       help='Path to fine-tuned checkpoint')
    
    # Sampling arguments
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampling instead of DDPM')
    parser.add_argument('--use_injection', action='store_true', default=True,
                       help='Use noise injection during sampling (RECOMMENDED, now default)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                       help='DDIM eta parameter (0.0 = deterministic)')
    parser.add_argument('--clip_denoised', action='store_true', default=True,
                       help='Clip denoised samples to [-1, 1]')
    parser.add_argument('--show_progress', action='store_true', default=True,
                       help='Show sampling progress bars')
    parser.add_argument('--blend_output', action='store_true', default=True,
                       help='Apply final blending')
    
    # Speed optimization arguments
    parser.add_argument('--fast_inference', action='store_true',
                       help='Enable fast inference mode (fewer steps, larger batches, quantization)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps (default: 50, original: 1000)')
    parser.add_argument('--use_half_precision', action='store_true',
                       help='Use FP16 for faster inference (requires modern GPU)')
    parser.add_argument('--skip_final_blend', action='store_true',
                       help='Skip final blending step for speed')
    parser.add_argument('--compile_model', action='store_true',
                       help='Compile model with PyTorch 2.0+ for speed')
    
    # Quantization arguments
    parser.add_argument('--quantize', choices=['none', 'simple', 'dynamic'], 
                       default='none', help='Quantization method')
    
    # Configuration
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for sampling')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--max_batches', type=int, default=0,
                       help='Maximum batches to process (0 = all)')
    
    # Random sampling
    parser.add_argument('--random_samples', type=int, default=100,
                       help='Number of random samples to select (default: 100, set to 0 to use all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    # Metrics
    parser.add_argument('--calculate_metrics', action='store_true', default=True,
                       help='Calculate FID, LPIPS, and SSIM metrics')
    
    # Debug options
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with single batch and reduced steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='inpainting_results',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Apply speed optimizations first
    args = apply_speed_optimizations(args)
    
    # Set train_dir default
    if args.train_dir is None:
        args.train_dir = args.test_dir
    
    # Quick test mode
    if args.quick_test:
        args.random_samples = 8
        args.max_batches = 2
        args.batch_size = 4
        args.ddim_steps = 10  # Very fast for testing
        print("Quick test mode: Using 8 samples, 2 batches, batch size 4, 10 DDIM steps")
    
    print(f"Configuration:")
    print(f"  Random samples: {'All' if args.random_samples == 0 else args.random_samples}")
    print(f"  Mask cycling: Ordered (will repeat if needed)")
    print(f"  Noise injection: {'Enabled' if args.use_injection else 'Disabled'}")
    print(f"  Sampling method: {'DDIM' if args.use_ddim or hasattr(args, 'ddim_steps') else 'DDPM'}")
    print(f"  DDIM steps: {getattr(args, 'ddim_steps', 'N/A')}")
    print(f"  Quantization: {args.quantize}")
    print(f"  Half precision: {'Enabled' if args.use_half_precision else 'Disabled'}")
    print(f"  Fast inference: {'Enabled' if args.fast_inference else 'Disabled'}")
    print(f"  Random seed: {args.seed}")
    
    # Create sampler and run
    sampler = InpaintingSampler(args)
    sampler.run_sampling()


if __name__ == "__main__":
    # Import PIL here to avoid issues
    from PIL import Image
    main()
