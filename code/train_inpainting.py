"""
Training utilities for diffusion inpainting models.
Updated to work with enhanced diffusion implementation with advanced inpainting injection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms


def train_epoch(model, dataloader, optimizer, diffusion, device, epoch=0):
    """Train with enhanced diffusion - uses advanced injection capabilities."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Track mask usage for monitoring
    mask_usage = {}

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        # Move data to device
        images = batch['image'].to(device)           # Target images [-1, 1]
        masked_images = batch['masked_image'].to(device)  # Images with holes
        masks = batch['mask'].to(device)             # Binary mask [0, 1]

        # Track mask usage (for first epoch only to avoid spam)
        if epoch == 0 and 'mask_path' in batch:
            for mask_path in batch['mask_path']:
                mask_name = Path(mask_path).name
                mask_usage[mask_name] = mask_usage.get(mask_name, 0) + 1

        # Sample random timesteps
        batch_size = images.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()

        # Model kwargs for inpainting (compatible with enhanced diffusion)
        model_kwargs = {
            'masked_image': masked_images, 
            'mask': masks
        }

        # Use enhanced diffusion's training_losses method (supports advanced injection)
        losses = diffusion.training_losses(
            model=model,
            x_start=images,
            t=t,
            model_kwargs=model_kwargs
        )
        
        loss = losses['loss']

        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.6f}',
            'mask_coverage': f'{masks.mean().item():.3f}'
        })
    


    return total_loss / num_batches


def validate(model, dataloader, diffusion, device):
    """
    Validate the model without visual sample generation.
    
    Args:
        model: The diffusion inpainting model
        dataloader: Validation data loader
        diffusion: Enhanced GaussianDiffusion object
        device: Device to validate on
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    total_mask_coverage = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            masked_images = batch['masked_image'].to(device)
            masks = batch['mask'].to(device)

            batch_size = images.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()
            
            # Model kwargs for validation
            model_kwargs = {
                'masked_image': masked_images,
                'mask': masks
            }
            
            # Use enhanced diffusion's training_losses method
            losses = diffusion.training_losses(
                model=model,
                x_start=images,
                t=t,
                model_kwargs=model_kwargs
            )
            
            loss = losses['loss']
            total_loss += loss.item()
            total_mask_coverage += masks.mean().item()

    avg_loss = total_loss / num_batches
    avg_coverage = total_mask_coverage / num_batches
    
    print(f"Validation - Loss: {avg_loss:.6f}, Avg Mask Coverage: {avg_coverage:.3f}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path, 
                   diffusion_config=None, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        val_loss: Validation loss
        save_path: Path to save checkpoint
        diffusion_config: Diffusion configuration dict
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'diffusion_config': diffusion_config or {}
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load on
        
    Returns:
        dict: Checkpoint information (epoch, val_loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'diffusion_config': checkpoint.get('diffusion_config', {})
    }


def create_model_and_diffusion(checkpoint_path, device, img_size=256):
    """
    Create model and enhanced diffusion from checkpoint.
    """
    # Import here to avoid circular imports
    from unet import UNetModel, DiffusionInpaintingModel
    from utils.schedules import create_gaussian_diffusion
    
    # Create base UNet model with the architecture from the original script
    base_model = UNetModel(
        image_size=img_size,
        in_channels=3,
        model_channels=128,
        out_channels=6,
        num_res_blocks=1,
        attention_resolutions=(16,),
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )
    
    # Load pretrained weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Load with flexible key matching
    missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    # Create inpainting model
    model = DiffusionInpaintingModel(base_model, in_channels=9).to(device)

    # Create enhanced diffusion process
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="quadratic",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
    )

    checkpoint_info = {
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys
    }
    
    return model, diffusion, checkpoint_info


def sample_with_advanced_inpainting(model, diffusion, masked_images, masks, device, 
                                   num_steps=50, use_ddim=True, eta=0.0, 
                                   injection_schedule="all", use_cumulative_noise=True):
    """
    Sample inpainting results using enhanced diffusion with advanced injection.
    
    Args:
        model: Inpainting diffusion model
        diffusion: Enhanced GaussianDiffusion object
        masked_images: Images with holes [-1, 1], shape [B, 3, H, W]
        masks: Binary mask [0, 1], shape [B, 1, H, W] where 1 = inpaint
        device: Device to sample on
        num_steps: Number of sampling steps (only used for custom sampling)
        use_ddim: Whether to use DDIM sampling
        eta: DDIM stochasticity (0.0 = deterministic)
        injection_schedule: When to apply injection ("all", "high", "low")
        use_cumulative_noise: Whether to use cumulative noise schedule
    
    Returns:
        torch.Tensor: Inpainted image [-1, 1]
    """
    model.eval()
    
    # Reconstruct ground truth from masked images (approximation)
    # In practice, you'd have the actual ground truth
    gt = masked_images / (1 - masks + 1e-8)  # Rough reconstruction
    gt = torch.clamp(gt, -1, 1)
    
    # Convert masks: input mask (1=inpaint) -> keep_mask (1=keep, 0=generate)
    gt_keep_mask = 1 - masks
    
    with torch.no_grad():
        result = diffusion.sample_with_advanced_inpainting(
            model=model,
            shape=masked_images.shape,
            gt=gt,
            gt_keep_mask=gt_keep_mask,
            use_ddim=use_ddim,
            eta=eta,
            progress=False,
            device=device,
            injection_schedule=injection_schedule,
            use_cumulative_noise=use_cumulative_noise
        )
    
    return result


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def create_cosine_scheduler(optimizer, num_epochs, warmup_epochs=0, min_lr_ratio=0.01):
    """
    Create cosine annealing scheduler with optional warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (default: 0)
        min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.01)
    
    Returns:
        torch.optim.lr_scheduler: Cosine annealing scheduler
    """
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Pure cosine annealing without warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs, 
            eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
        )
    
    return scheduler


def create_optimizer_and_scheduler(model, lr, weight_decay, num_epochs, 
                                 warmup_epochs=0, scheduler_type='cosine', min_lr_ratio=0.01):
    """
    Create optimizer and scheduler with support for cosine annealing.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay
        num_epochs: Total training epochs
        warmup_epochs: Warmup epochs (default: 0)
        scheduler_type: 'cosine', 'step', or 'none'
        min_lr_ratio: Minimum LR ratio for cosine scheduling
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    if scheduler_type == 'cosine':
        scheduler = create_cosine_scheduler(
            optimizer, num_epochs, warmup_epochs, min_lr_ratio
        )
        print(f"Created cosine scheduler with warmup_epochs={warmup_epochs}, min_lr_ratio={min_lr_ratio}")
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.5)
        print("Created step scheduler")
    else:
        scheduler = None
        print("No scheduler created")
    
    return optimizer, scheduler


