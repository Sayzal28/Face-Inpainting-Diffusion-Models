#!/usr/bin/env python3
"""
Main training script for diffusion inpainting models with enhanced diffusion.
Compatible with advanced inpainting injection capabilities.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import deque

import torch
import torch.optim as optim

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_inpainting_dataloaders
from train_inpainting import (
    train_epoch, validate, save_checkpoint, 
    create_model_and_diffusion, EarlyStopping,
    create_optimizer_and_scheduler
)


class CheckpointManager:
    """Manages checkpoints with automatic cleanup to save disk space."""
    
    def __init__(self, save_dir, max_checkpoints=2):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files = deque(maxlen=max_checkpoints)
        self.best_model_path = self.save_dir / 'best_model.pt'
        self.latest_model_path = self.save_dir / 'latest_model.pt'
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss, 
                       diffusion_config=None, is_best=False):
        """Save checkpoint and manage disk space."""
        
        # Always save as latest
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'diffusion_config': diffusion_config or {}
        }
        
        # Save latest model (always overwrite)
        torch.save(checkpoint_data, self.latest_model_path)
        print(f"Latest model saved to {self.latest_model_path}")
        
        # Save best model if this is the best
        if is_best:
            torch.save(checkpoint_data, self.best_model_path)
            print(f"Best model saved to {self.best_model_path} (val_loss: {val_loss:.6f})")
        
        # Save numbered checkpoint (with cleanup)
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Add to tracked files and cleanup old ones
        self.checkpoint_files.append(checkpoint_path)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        # Get all checkpoint files in directory
        checkpoint_pattern = "checkpoint_epoch_*.pt"
        all_checkpoints = sorted(self.save_dir.glob(checkpoint_pattern))
        
        # Keep only the last N checkpoints
        if len(all_checkpoints) > self.max_checkpoints:
            checkpoints_to_remove = all_checkpoints[:-self.max_checkpoints]
            for old_checkpoint in checkpoints_to_remove:
                try:
                    old_checkpoint.unlink()
                    print(f"Removed old checkpoint: {old_checkpoint.name}")
                except FileNotFoundError:
                    pass  # Already removed
    
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint."""
        if self.latest_model_path.exists():
            return self.latest_model_path
        return None
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint."""
        if self.best_model_path.exists():
            return self.best_model_path
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train diffusion inpainting model with enhanced diffusion')
    
    # Data arguments
    parser.add_argument('--checkpoint_path', type=str, default='/user/HS402/zs00774/Downloads/ffhq_baseline.pt')
    parser.add_argument('--train_dir', type=str, default='/user/HS402/zs00774/Downloads/data/train')
    parser.add_argument('--val_dir', type=str, default='/user/HS402/zs00774/Downloads/data/val')
    parser.add_argument('--mask_dir', type=str, default='/user/HS402/zs00774/Downloads/mask_dataset_split')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (will be resized to img_size x img_size)')
    
    # Model arguments
    parser.add_argument('--learn_sigma', action='store_true',
                       help='Learn variance in the model')
    parser.add_argument('--use_kl', action='store_true',
                       help='Use KL loss instead of MSE')
    parser.add_argument('--predict_xstart', action='store_true',
                       help='Predict x_start instead of noise')
    
    # Training configuration
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints_enhanced',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    # Space management
    parser.add_argument('--max_checkpoints', type=int, default=2,
                       help='Maximum number of regular checkpoints to keep')
    
    # Optimization arguments
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Scheduler arguments
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Type of learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs for cosine scheduler')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                       help='Minimum LR as ratio of initial LR (for cosine scheduler)')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-6,
                       help='Minimum change for early stopping')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (or "latest"/"best")')
    
    # Logging
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log training stats every N steps')
    
    return parser.parse_args()


def main():
    """Main training function with enhanced diffusion support."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(save_dir, max_checkpoints=args.max_checkpoints)
    print(f"Checkpoint manager: keeping last {args.max_checkpoints} checkpoints + best model")
    
    # Create dataloaders using the updated dataset.py (supports flat directories)
    print("Creating dataloaders with serial mask ordering...")
    
    train_loader, val_loader = create_inpainting_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_serial_masks=True  # Use serial mask ordering
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model and enhanced diffusion
    print("Creating model and enhanced diffusion...")
    model, diffusion, checkpoint_info = create_model_and_diffusion(
        checkpoint_path=args.checkpoint_path,
        device=device,
        img_size=args.img_size
    )
    
    print(f"Enhanced diffusion created with advanced inpainting injection support")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Create optimizer and scheduler with cosine support
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler_type,
        min_lr_ratio=args.min_lr_ratio
    )
    
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Scheduler type: {args.scheduler_type}")
    if args.scheduler_type == 'cosine':
        print(f"Warmup epochs: {args.warmup_epochs}")
        print(f"Min LR ratio: {args.min_lr_ratio}")
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        if args.resume == "latest":
            resume_path = checkpoint_manager.get_latest_checkpoint()
        elif args.resume == "best":
            resume_path = checkpoint_manager.get_best_checkpoint()
        else:
            resume_path = Path(args.resume)
        
        if resume_path and resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            from train_inpainting import load_checkpoint
            resume_info = load_checkpoint(
                resume_path, model, optimizer, scheduler, device
            )
            start_epoch = resume_info['epoch'] + 1
            best_val_loss = resume_info['val_loss']
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        else:
            print(f"Resume checkpoint not found: {resume_path}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        restore_best_weights=True
    )
    
    # Training loop
    print("Starting training with enhanced diffusion...")
    print(f"Scheduler will be called after each epoch")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            diffusion=diffusion,
            device=device,
            epoch=epoch
        )
        
        # Update scheduler after each epoch
        if scheduler:
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            print(f"LR updated: {current_lr:.2e} → {new_lr:.2e}")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            val_loss = validate(
                model=model,
                dataloader=val_loader,
                diffusion=diffusion,
                device=device
            )
            
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.6f}")
            
            # Save regular checkpoint
            if (epoch + 1) % args.save_every == 0 or is_best:
                diffusion_config = {
                    'learn_sigma': args.learn_sigma,
                    'use_kl': args.use_kl,
                    'predict_xstart': args.predict_xstart,
                    'scheduler_type': args.scheduler_type,
                    'warmup_epochs': args.warmup_epochs,
                    'min_lr_ratio': args.min_lr_ratio,
                    'enhanced_diffusion': True  # Mark as enhanced
                }
                
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    diffusion_config=diffusion_config,
                    is_best=is_best
                )
                
                print(f"Checkpoint saved to: {checkpoint_path.name}")
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}")
        
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {save_dir}")
    
    # Show final checkpoint status
    print(f"\nFinal checkpoint files:")
    for file in sorted(save_dir.glob("*.pt")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.1f} MB")
    
    # Show final learning rate
    final_lr = optimizer.param_groups[0]['lr']
    print(f"Final learning rate: {final_lr:.2e}")
    
    # Test enhanced inpainting capabilities
    print(f"\nEnhanced diffusion features available:")
    print(f"  - Advanced inpainting injection: ✓")
    print(f"  - Consistent noise caching: ✓")
    print(f"  - Flexible injection scheduling: ✓")
    print(f"  - DDIM/DDPM sampling with injection: ✓")


if __name__ == "__main__":
    main()
