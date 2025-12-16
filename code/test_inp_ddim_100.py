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
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_inference_dataloader, OrderedMaskDataset, FlatImageDataset
from train_inpainting import create_model_and_diffusion

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
    print("pytorch-fid loaded successfully")
except ImportError:
    print("Warning: pytorch-fid not installed. FID calculation will be skipped.")
    print("Install with: pip install pytorch-fid")
    FID_AVAILABLE = False

def toU8(sample):
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

    # Save comparison grid
    comparison_list = []
    if gts is not None:
        comparison_list.append(torch.from_numpy(gts).permute(0, 3, 1, 2) / 127.5 - 1)
    if masked_imgs is not None:
        comparison_list.append(torch.from_numpy(masked_imgs).permute(0, 3, 1, 2) / 127.5 - 1)
    if srs is not None:
        comparison_list.append(torch.from_numpy(srs).permute(0, 3, 1, 2) / 127.5 - 1)
    
    if comparison_list:
        # Reorganize tensors for proper horizontal layout
        # We want: [orig1, masked1, inpainted1, orig2, masked2, inpainted2, ...]
        reorganized_images = []
        
        for i in range(batch_size):
            for comparison_type in comparison_list:
                reorganized_images.append(comparison_type[i:i+1])  # Take single image
        
        comparison = torch.cat(reorganized_images, dim=0)
        
        # Set nrow to number of comparison types (typically 3)
        # This creates: batch_size rows, len(comparison_list) columns
        # Layout: [orig1, masked1, inpainted1]
        #         [orig2, masked2, inpainted2]
        #         [orig3, masked3, inpainted3] ...
        nrow = len(comparison_list)
        
        save_image(
            comparison,
            output_dir / f"comparison_batch_{batch_idx}.png",
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1)
        )


class MetricsCalculator:
    """
    Calculate various image quality metrics with proper implementations.
    Replaces FID with a more robust torchmetrics and cleanfid approach.
    """
    
    def __init__(self, device):
        self.device = device
        
        # Initialize LPIPS only once
        print("Initializing LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        print("LPIPS model loaded successfully")
        
        # Storage for metric calculation - store on CPU to save GPU memory
        self.all_original = []
        self.all_inpainted = []
        print(f"MetricsCalculator initialized. pytorch-fid available: {FID_AVAILABLE}")
    
    def add_batch(self, original_images, inpainted_images):
        """Add a batch of images for metric calculation."""
        # Keep in [-1, 1] range and store on CPU immediately
        self.all_original.append(original_images.cpu())
        self.all_inpainted.append(inpainted_images.cpu())
    
    def calculate_lpips_batch(self, img1, img2):
        """Calculate LPIPS for a batch - expects [-1, 1] range."""
        with torch.no_grad():
            lpips_dist = self.lpips_fn(img1, img2)
            return lpips_dist.cpu().numpy()
    
    def calculate_ssim_batch(self, img1, img2):
        """Calculate SSIM for a batch with proper parameters."""
        # Convert to [0, 1] range and numpy
        img1_np = ((img1.cpu() + 1) / 2).clamp(0, 1).numpy()
        img2_np = ((img2.cpu() + 1) / 2).clamp(0, 1).numpy()
        
        ssim_scores = []
        for i in range(img1_np.shape[0]):
            # Convert from CHW to HWC for SSIM
            img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
            img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
            
            score = ssim(
                img1_hwc,
                img2_hwc,
                channel_axis=2,
                data_range=1.0,
                win_size=11
            )
            ssim_scores.append(score)
        
        return np.array(ssim_scores)
    
    def save_images_for_fid(self, output_dir):
        """Save images in format required for pytorch-fid calculation."""
        if not FID_AVAILABLE:
            print("pytorch-fid not available")
            return None, None
            
        # Create directories for FID
        real_dir = Path(output_dir) / "fid_real"
        fake_dir = Path(output_dir) / "fid_generated"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all images
        all_real = torch.cat(self.all_original, dim=0)
        all_fake = torch.cat(self.all_inpainted, dim=0)
        
        print(f"Saving {len(all_real)} images for pytorch-fid calculation...")
        
        # Convert from [-1, 1] to [0, 1] range for saving
        all_real_01 = ((all_real + 1) / 2).clamp(0, 1)
        all_fake_01 = ((all_fake + 1) / 2).clamp(0, 1)
        
        # Save images with proper format
        for i in tqdm(range(len(all_real)), desc="Saving images for FID"):
            # Convert to PIL and save as PNG
            real_pil = transforms.ToPILImage()(all_real_01[i])
            fake_pil = transforms.ToPILImage()(all_fake_01[i])
            
            real_pil.save(real_dir / f"real_{i:04d}.png")
            fake_pil.save(fake_dir / f"fake_{i:04d}.png")
        
        print(f"Saved {len(all_real)} real images to {real_dir}")
        print(f"Saved {len(all_fake)} fake images to {fake_dir}")
        
        return str(real_dir), str(fake_dir)
    
    def calculate_fid(self, real_dir, fake_dir):
        """Calculate FID score using pytorch-fid."""
        if not FID_AVAILABLE:
            print("pytorch-fid not available")
            return None
            
        try:
            print("Calculating FID score with pytorch-fid...")
            print(f"Real images directory: {real_dir}")
            print(f"Fake images directory: {fake_dir}")
            
            # Use pytorch-fid with proper parameters
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=50,  # Reasonable batch size
                device=self.device,
                dims=2048,      # Standard InceptionV3 feature dimension
                num_workers=1   # Avoid multiprocessing issues
            )
            
            print(f"FID score calculated: {fid_value}")
            return round(fid_value, 4)
            
        except Exception as e:
            print(f"Error calculating FID with pytorch-fid: {e}")
            import traceback
            traceback.print_exc()
            return None


    def calculate_all_metrics(self, output_dir):
        """Calculate all metrics with proper implementations."""
        if not self.all_original or not self.all_inpainted:
            print("No images available for metric calculation")
            return {}
        
        # Concatenate all images
        all_real = torch.cat(self.all_original, dim=0)
        all_fake = torch.cat(self.all_inpainted, dim=0)
        
        print(f"\nCalculating metrics for {len(all_real)} images...")
        results = {}
        
        # Calculate LPIPS
        print("Calculating LPIPS...")
        lpips_scores = []
        batch_size = 16
        
        for i in range(0, len(all_real), batch_size):
            end_idx = min(i + batch_size, len(all_real))
            batch_real = all_real[i:end_idx].to(self.device)
            batch_fake = all_fake[i:end_idx].to(self.device)
            
            try:
                batch_lpips = self.calculate_lpips_batch(batch_real, batch_fake)
                lpips_scores.extend(batch_lpips.flatten())
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory error at batch {i//batch_size}, skipping...")
                    continue
                else:
                    raise e
            finally:
                del batch_real, batch_fake
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if lpips_scores:
            results['LPIPS'] = np.mean(lpips_scores)
        
        # Calculate SSIM
        print("Calculating SSIM...")
        ssim_scores = []
        for i in range(0, len(all_real), batch_size):
            end_idx = min(i + batch_size, len(all_real))
            batch_real = all_real[i:end_idx]
            batch_fake = all_fake[i:end_idx]
            
            try:
                batch_ssim = self.calculate_ssim_batch(batch_real, batch_fake)
                ssim_scores.extend(batch_ssim)
            except Exception as e:
                print(f"SSIM calculation error at batch {i//batch_size}: {e}")
                continue
        
        if ssim_scores:
            results['SSIM'] = np.mean(ssim_scores)
        
        # Calculate FID using pytorch-fid
        if FID_AVAILABLE:
            print("\nPreparing images for pytorch-fid...")
            real_dir, fake_dir = self.save_images_for_fid(output_dir)
            if real_dir and fake_dir:
                fid_value = self.calculate_fid(real_dir, fake_dir)
                if fid_value is not None:
                    results['FID'] = fid_value
        else:
            print("pytorch-fid calculation skipped - library not available")
        
        return results

class InpaintingSampler:
    """Professional inpainting sampler with noise injection """
    
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
        """Load standard model."""
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
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _create_dataloader(self):
        """Create dataloader for inpainting with ordered mask cycling."""
        print("Creating dataloader with ordered mask cycling...")
        
        # Use the convenient function from dataset.py
        self.dataloader = create_inference_dataloader(
            test_dir=self.args.test_dir,
            mask_dir=self.args.mask_dir,
            split='test',
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
    
    def create_ddim_timestep_sequence(self, total_timesteps, ddim_timesteps):
        """Create proper DDIM timestep sequence for acceleration."""
        # Create evenly spaced timesteps
        c = total_timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, total_timesteps, c)))
        
        # Add the last timestep if not included
        if ddim_timestep_seq[-1] != total_timesteps - 1:
            ddim_timestep_seq = np.append(ddim_timestep_seq, total_timesteps - 1)
        
        # Reverse for sampling (high to low noise)
        ddim_timestep_seq = ddim_timestep_seq[::-1]
        
        return ddim_timestep_seq
    
    def inpainting_p_sample_loop(self, model_fn, shape, gt_images, masks,
                                 clip_denoised=True, device=None, progress=False):
        """
        DDPM sampling with inpainting noise injection.
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        assert isinstance(shape, (tuple, list))
        
        # Start with pure noise
        img = torch.randn(*shape, device=device)
        
        # Get timestep indices in reverse order (1000 → 0)
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            indices = tqdm(indices, desc="DDPM inpainting with injection")
        
        # Prepare masks and known regions
        gt_keep_mask = 1 - masks
        
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
                # Replace known regions with properly noised original image
                if i > 0:
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
    
    def inpainting_ddim_sample_loop(self, model_fn, shape, gt_images, masks,
                                   clip_denoised=True, device=None, progress=False, eta=0.0):
        """
        CORRECTED DDIM sampling with inpainting noise injection and proper acceleration.
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        assert isinstance(shape, (tuple, list))
        
        # Start with pure noise
        img = torch.randn(*shape, device=device)
        
        # FIXED: Create subset of timesteps for DDIM acceleration
        total_timesteps = self.diffusion.num_timesteps
        ddim_timesteps = self.args.ddim_timesteps
        ddim_timestep_seq = self.create_ddim_timestep_sequence(total_timesteps, ddim_timesteps)
        
        print(f"DDIM acceleration: Using {ddim_timesteps} steps instead of {total_timesteps}")
        print(f"Timestep sequence: {ddim_timestep_seq[:5]}...{ddim_timestep_seq[-5:]}")
        
        if progress:
            timestep_iterator = tqdm(enumerate(ddim_timestep_seq),
                                   desc=f"DDIM inpainting ({ddim_timesteps} steps)",
                                   total=len(ddim_timestep_seq))
        else:
            timestep_iterator = enumerate(ddim_timestep_seq)
        
        # Prepare masks and known regions
        gt_keep_mask = 1 - masks
        
        for step_idx, timestep in timestep_iterator:
            t = torch.tensor([timestep] * shape[0], device=device)
            
            with torch.no_grad():
                # STEP 1: DDIM step with corrected formula
                model_kwargs = {
                    "gt": gt_images,
                    "gt_keep_mask": gt_keep_mask,
                }
                
                # Get model output - may be noise only or noise + variance
                model_output = model_fn(img, t, **model_kwargs)
                
                # FIXED: Handle model that outputs both noise and variance
                if model_output.shape[1] == 6:
                    noise_pred = model_output[:, :3]
                elif model_output.shape[1] == 3:
                    noise_pred = model_output
                else:
                    raise ValueError(f"Unexpected model output shape: {model_output.shape}")
                
                # FIXED: Proper DDIM update formula
                alpha_t = self.diffusion.alphas_cumprod[timestep]
                
                # Get previous timestep for DDIM
                if step_idx < len(ddim_timestep_seq) - 1:
                    prev_timestep = ddim_timestep_seq[step_idx + 1]
                    alpha_prev = self.diffusion.alphas_cumprod[prev_timestep]
                else:
                    prev_timestep = 0
                    alpha_prev = 1.0
                
                alpha_t = torch.tensor(alpha_t, device=device)
                alpha_prev = torch.tensor(alpha_prev, device=device)
                
                # Predict x_0 from noise prediction
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                pred_x0 = (img - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                
                if clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                # Calculate direction to x_t
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                
                # DDIM variance
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
                
                # DDIM deterministic direction
                pred_dir = torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred
                
                # Sample noise (zero for deterministic DDIM when eta=0)
                noise = torch.randn_like(img) if timestep > 0 and eta > 0 else torch.zeros_like(img)
                
                # DDIM update
                img = sqrt_alpha_prev * pred_x0 + pred_dir + sigma * noise
                
                # STEP 2: CORRECTED INPAINTING INJECTION
                if timestep > 0:
                    # Use the correct alpha for noise injection
                    injection_alpha = alpha_prev
                    sqrt_alpha_inject = torch.sqrt(injection_alpha)
                    sqrt_one_minus_alpha_inject = torch.sqrt(1 - injection_alpha)
                    
                    # Add correct noise level to known regions
                    known_noise = torch.randn_like(gt_images)
                    noised_known_regions = (
                        sqrt_alpha_inject * gt_images +
                        sqrt_one_minus_alpha_inject * known_noise
                    )
                    
                    # Inject: keep generated content in holes, use noised original in known regions
                    img = img * masks + noised_known_regions * gt_keep_mask
        
        return img

    def sample_batch(self, batch):
        """Sample inpainting for a batch with noise injection."""
        # Move batch to device
        gt_images = batch['image'].to(self.device)
        masked_images = batch['masked_image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        batch_size = gt_images.shape[0]
        
        # Print mask info for first batch
        if not hasattr(self, '_first_batch_printed'):
            self._first_batch_printed = True
            if 'mask_path' in batch:
                print(f"Using masks: {[Path(p).name for p in batch['mask_path']]}")
            if 'mask_idx' in batch:
                print(f"Mask indices: {batch['mask_idx'].tolist()}")
        
        print(f"Sampling batch of {batch_size} images...")
        print(f"GT images shape: {gt_images.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Device: {self.device}")
        
        # Test model function first
        print("Testing model function...")
        try:
            with torch.no_grad():
                test_x = torch.randn(1, 3, self.args.img_size, self.args.img_size).to(self.device)
                test_t = torch.tensor([100]).to(self.device)
                test_kwargs = {
                    "gt": gt_images[:1],
                    "gt_keep_mask": (1 - masks[:1])
                }
                test_output = self.model_fn(test_x, test_t, **test_kwargs)
                print(f"Model test successful. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Model function test failed: {e}")
            raise
        
        # Choose sampling method with injection
        if self.args.use_injection:
            if self.args.use_ddim:
                sampling_method = f"DDIM with injection ({self.args.ddim_timesteps} steps)"
                print(f"Using {sampling_method}...")
                
                # Use corrected DDIM sampling with injection
                result = self.inpainting_ddim_sample_loop(
                    self.model_fn,
                    (batch_size, 3, self.args.img_size, self.args.img_size),
                    gt_images,
                    masks,
                    clip_denoised=self.args.clip_denoised,
                    device=self.device,
                    progress=self.args.show_progress,
                    eta=self.args.ddim_eta
                )
            else:
                sampling_method = "DDPM with injection"
                print(f"Using {sampling_method}...")
                
                # Use custom DDPM sampling with injection
                result = self.inpainting_p_sample_loop(
                    self.model_fn,
                    (batch_size, 3, self.args.img_size, self.args.img_size),
                    gt_images,
                    masks,
                    clip_denoised=self.args.clip_denoised,
                    device=self.device,
                    progress=self.args.show_progress
                )
        else:
            # Original sampling without injection
            model_kwargs = {
                "gt": gt_images,
                "gt_keep_mask": 1 - masks,
            }
            
            if self.args.use_ddim:
                if hasattr(self.diffusion, 'ddim_sample_loop'):
                    sample_fn = self.diffusion.ddim_sample_loop
                    sampling_method = "DDIM (original)"
                else:
                    print("Warning: DDIM not available, using DDPM sampling")
                    sample_fn = self.diffusion.p_sample_loop
                    sampling_method = "DDPM (original fallback)"
            else:
                sample_fn = self.diffusion.p_sample_loop
                sampling_method = "DDPM (original)"
            
            print(f"Using {sampling_method}...")
            print(f"Diffusion timesteps: {self.diffusion.num_timesteps}")
            
            print("Starting sampling loop...")
            with torch.no_grad():
                if self.args.use_ddim and hasattr(self.diffusion, 'ddim_sample_loop'):
                    result = sample_fn(
                        self.model_fn,
                        (batch_size, 3, self.args.img_size, self.args.img_size),
                        clip_denoised=self.args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=self.device,
                        progress=self.args.show_progress
                    )
                else:
                    result = sample_fn(
                        self.model_fn,
                        (batch_size, 3, self.args.img_size, self.args.img_size),
                        clip_denoised=self.args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=self.device,
                        progress=self.args.show_progress
                    )
        
        print("Sampling completed!")
        
        # Final blending (optional with injection, required without)
        if self.args.blend_output:
            print("Applying final blending...")
            keep_mask = 1 - masks
            result = result * masks + gt_images * keep_mask
        
        return result, gt_images, masked_images, masks
    
    def run_sampling(self):
        """Run complete sampling process."""
        injection_text = " with injection" if self.args.use_injection else " (original)"
        ddim_text = f" ({self.args.ddim_timesteps} steps)" if self.args.use_ddim else ""
        print(f"Starting inpainting sampling{injection_text}...")
        print(f"Sampling method: {'DDIM' if self.args.use_ddim else 'DDPM'}{ddim_text}{injection_text}")
        print(f"Output directory: {self.output_dir}")
        
        total_time = 0
        total_samples = 0
        
        max_batches = self.args.max_batches if self.args.max_batches > 0 else len(self.dataloader)
        
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing batches")):
            if batch_idx >= max_batches:
                break
            
            start_time = time.time()
            
            # Sample batch
            sampled, gt_images, masked_images, masks = self.sample_batch(batch)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += gt_images.shape[0]
            
            # Add to metrics calculator immediately to save GPU memory
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
            
            # FIXED: Clear GPU memory after each batch
            del sampled, gt_images, masked_images, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Batch {batch_idx + 1}/{max_batches} completed in {batch_time:.2f}s")
        
        avg_time = total_time / total_samples if total_samples > 0 else 0
        print(f"\nSampling completed!")
        print(f"Total samples: {total_samples}")
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
                print(f"{metric_name:25s}: {value:.4f}")
            print("="*50)
            
            # Save metrics to file
            metrics_file = self.output_dir / "metrics.txt"
            injection_text = " with injection" if self.args.use_injection else " (original)"
            ddim_text = f" ({self.args.ddim_timesteps} steps)" if self.args.use_ddim else ""
            
            with open(metrics_file, 'w') as f:
                f.write(f"Evaluation Metrics\n")
                f.write(f"==================\n")
                f.write(f"Total samples: {total_samples}\n")
                f.write(f"Total time: {total_time:.2f}s\n")
                f.write(f"Sampling method: {'DDIM' if self.args.use_ddim else 'DDPM'}{ddim_text}{injection_text}\n")
                f.write(f"Avg time per sample: {avg_time:.2f}s\n\n")
                
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            print(f"Metrics saved to: {metrics_file}")
        else:
            print("Metrics calculation skipped (use --calculate_metrics to enable)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inpainting sampler with optional noise injection and metrics')
    
    # Data arguments
    parser.add_argument('--test_dir', type=str, default='/user/HS402/zs00774/Downloads/data/test',
                       help='Directory containing test images')
    parser.add_argument('--mask_dir', type=str, default='/user/HS402/zs00774/Downloads/mask_dataset_split',
                       help='Directory containing mask images')
    parser.add_argument('--train_dir', type=str, default='/user/HS402/zs00774/Downloads/data/train',
                       help='Train directory (dummy, required by dataloader)')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, default='/user/HS402/zs00774/Downloads/ffhq_baseline.pt',
                       help='Path to base model checkpoint')
    parser.add_argument('--trained_checkpoint', type=str, default='/user/HS402/zs00774/Downloads/idkyet_ffid/checkpoints/best_model.pt',
                       help='Path to fine-tuned checkpoint')
    
    # Sampling arguments
    parser.add_argument('--use_ddim', action='store_true',default=True,
                       help='Use DDIM sampling instead of DDPM')
    parser.add_argument('--ddim_timesteps', type=int, default=50,
                       help='Number of DDIM timesteps (default: 50 for acceleration)')
    parser.add_argument('--use_injection', action='store_true', default=True,
                       help='Use noise injection during sampling (RECOMMENDED, now default)')
    parser.add_argument('--ddim_eta', type=float, default=0.75,
                       help='DDIM eta parameter (0.0 = deterministic, >0 = stochastic)')
    parser.add_argument('--clip_denoised', action='store_true', default=True,
                       help='Clip denoised samples to [-1, 1]')
    parser.add_argument('--show_progress', action='store_true', default=True,
                       help='Show sampling progress bars')
    parser.add_argument('--blend_output', action='store_true', default=True,
                       help='Apply final blending')
    
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
    parser.add_argument('--random_samples', type=int, default=0,
                       help='Number of random samples to select (default: 0 = use all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    # Metrics
    parser.add_argument('--calculate_metrics', action='store_true', default=True,
                       help='Calculate FID, LPIPS, and SSIM metrics')
    
    # Debug options
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with single batch and reduced steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='ddim_50_eta_0.75',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set train_dir default
    if args.train_dir is None:
        args.train_dir = args.test_dir
    
    # Quick test mode
    if args.quick_test:
        args.random_samples = 8
        args.max_batches = 2
        args.batch_size = 4
        args.ddim_timesteps = 10
        print("Quick test mode: Using 8 samples, 2 batches, batch size 4, 10 DDIM steps")
    
    print(f"Configuration:")
    print(f"  Random samples: {'All' if args.random_samples == 0 else args.random_samples}")
    print(f"  Mask cycling: Ordered (will repeat if needed)")
    print(f"  Noise injection: {'Enabled' if args.use_injection else 'Disabled'}")
    print(f"  Sampling method: {'DDIM' if args.use_ddim else 'DDPM'}")
    if args.use_ddim:
        print(f"  DDIM timesteps: {args.ddim_timesteps} (acceleration: {1000//args.ddim_timesteps}x)")
        print(f"  DDIM eta: {args.ddim_eta} ({'deterministic' if args.ddim_eta == 0.0 else 'stochastic'})")
    print(f"  Random seed: {args.seed}")
    
    # Create sampler and run
    sampler = InpaintingSampler(args)
    sampler.run_sampling()


if __name__ == "__main__":
    main()
