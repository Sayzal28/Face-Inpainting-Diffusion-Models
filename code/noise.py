#!/usr/bin/env python3
"""
Script to visualize how different noise schedules (linear, cosine, quadratic) 
add noise to clean images at specific timesteps during the forward diffusion process.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Linear noise schedule."""
    return np.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule from Improved DDPM paper."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def quadratic_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Quadratic noise schedule."""
    return np.linspace(start**0.5, end**0.5, timesteps) ** 2


class NoiseScheduleVisualizer:
    """Visualize forward diffusion process for different noise schedules."""
    
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the three noise schedules
        self.schedules = {
            'linear': linear_beta_schedule(timesteps),
            'cosine': cosine_beta_schedule(timesteps),
            'quadratic': quadratic_beta_schedule(timesteps)
        }
        
        # Calculate diffusion parameters for each schedule
        self.diffusion_params = {}
        for name, betas in self.schedules.items():
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas)
            
            self.diffusion_params[name] = {
                'betas': betas,
                'alphas': alphas,
                'alphas_cumprod': alphas_cumprod,
                'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
                'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod)
            }
    
    def add_noise_at_timestep(self, x0, timestep, schedule_name, noise=None):
        """Add noise to clean image x0 at specific timestep using given schedule."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        params = self.diffusion_params[schedule_name]
        
        # Get noise parameters at this timestep
        sqrt_alpha_cumprod = params['sqrt_alphas_cumprod'][timestep]
        sqrt_one_minus_alpha_cumprod = params['sqrt_one_minus_alphas_cumprod'][timestep]
        
        # Apply forward diffusion: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        xt = (sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise)
        
        return xt
    
    def visualize_noise_progression(self, image_path, timesteps_to_show, output_dir):
        """Create visualization showing noise progression across timesteps."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        
        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        x0 = transform(image).unsqueeze(0).to(self.device)
        
        # Generate consistent noise for all schedules
        torch.manual_seed(42)  # For reproducible results
        base_noise = torch.randn_like(x0)
        
        # Store results for each schedule
        results = {}
        
        for schedule_name in self.schedules.keys():
            print(f"Processing {schedule_name} schedule...")
            
            schedule_results = []
            schedule_results.append(x0.clone())  # Add clean image
            
            for t in timesteps_to_show:
                noisy_image = self.add_noise_at_timestep(
                    x0, t, schedule_name, noise=base_noise
                )
                schedule_results.append(noisy_image)
            
            results[schedule_name] = schedule_results
        
        # Create comparison grids
        self._save_schedule_comparison(results, timesteps_to_show, output_dir)
        self._save_timestep_comparison(results, timesteps_to_show, output_dir)
        self._save_individual_progressions(results, timesteps_to_show, output_dir)
        
        # Save numerical analysis
        self._save_numerical_analysis(timesteps_to_show, output_dir)
        
        print(f"Visualization saved to {output_dir}")
    
    def _save_schedule_comparison(self, results, timesteps_to_show, output_dir):
        """Save grid comparing all schedules side by side."""
        # Create grid: rows = schedules, columns = timesteps (including t=0)
        all_images = []
        
        for schedule_name in ['linear', 'cosine', 'quadratic']:
            schedule_images = results[schedule_name]
            all_images.extend(schedule_images)
        
        # Convert to tensor
        comparison_tensor = torch.cat(all_images, dim=0)
        
        # Save grid
        nrow = len(timesteps_to_show) + 1  # +1 for clean image
        save_image(
            comparison_tensor,
            output_dir / 'noise_schedule_comparison.png',
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
            padding=4,
            pad_value=1.0
        )
        
        # Create labeled version with matplotlib
        self._create_labeled_comparison(results, timesteps_to_show, output_dir)
    
    def _create_labeled_comparison(self, results, timesteps_to_show, output_dir):
        """Create a labeled comparison grid with titles."""
        fig_width = 3 * (len(timesteps_to_show) + 1)  # +1 for clean image
        fig_height = 3 * 3  # 3 schedules
        
        fig, axes = plt.subplots(3, len(timesteps_to_show) + 1, figsize=(fig_width, fig_height))
        
        schedule_names = ['linear', 'cosine', 'quadratic']
        timestep_labels = ['Clean (t=0)'] + [f't={t}' for t in timesteps_to_show]
        
        for row, schedule_name in enumerate(schedule_names):
            for col, (img_tensor, label) in enumerate(zip(results[schedule_name], timestep_labels)):
                ax = axes[row, col]
                
                # Convert tensor to numpy for plotting
                img_np = img_tensor.squeeze().cpu().numpy()
                img_np = (img_np + 1) / 2  # Convert from [-1,1] to [0,1]
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
                img_np = np.clip(img_np, 0, 1)
                
                ax.imshow(img_np)
                ax.set_title(f'{schedule_name.title()}\n{label}', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'labeled_noise_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_timestep_comparison(self, results, timesteps_to_show, output_dir):
        """Save comparison at each timestep across all schedules."""
        for i, t in enumerate(timesteps_to_show):
            timestep_images = []
            
            # Add images from each schedule at this timestep (+1 to skip clean image)
            for schedule_name in ['linear', 'cosine', 'quadratic']:
                timestep_images.append(results[schedule_name][i + 1])
            
            timestep_tensor = torch.cat(timestep_images, dim=0)
            
            save_image(
                timestep_tensor,
                output_dir / f'timestep_{t:04d}_comparison.png',
                nrow=3,
                normalize=True,
                value_range=(-1, 1),
                padding=2,
                pad_value=1.0
            )
    
    def _save_individual_progressions(self, results, timesteps_to_show, output_dir):
        """Save individual progression for each schedule."""
        progressions_dir = output_dir / 'individual_progressions'
        progressions_dir.mkdir(exist_ok=True)
        
        for schedule_name in self.schedules.keys():
            schedule_tensor = torch.cat(results[schedule_name], dim=0)
            
            save_image(
                schedule_tensor,
                progressions_dir / f'{schedule_name}_progression.png',
                nrow=len(timesteps_to_show) + 1,
                normalize=True,
                value_range=(-1, 1),
                padding=2,
                pad_value=1.0
            )
    
    def _save_numerical_analysis(self, timesteps_to_show, output_dir):
        """Save numerical analysis of noise parameters at each timestep."""
        analysis_file = output_dir / 'noise_analysis.txt'
        
        with open(analysis_file, 'w') as f:
            f.write("Noise Schedule Analysis at Specific Timesteps\n")
            f.write("=" * 60 + "\n\n")
            
            # Header
            f.write(f"{'Timestep':<10} {'Schedule':<12} {'Signal':<10} {'Noise':<10} {'SNR':<10}\n")
            f.write("-" * 60 + "\n")
            
            for t in [0] + timesteps_to_show:
                for schedule_name in self.schedules.keys():
                    if t == 0:
                        signal, noise, snr = 1.0, 0.0, float('inf')
                    else:
                        params = self.diffusion_params[schedule_name]
                        signal = params['sqrt_alphas_cumprod'][t]
                        noise = params['sqrt_one_minus_alphas_cumprod'][t]
                        snr = signal / (noise + 1e-8)
                    
                    f.write(f"{t:<10} {schedule_name:<12} {signal:<10.6f} {noise:<10.6f} {snr:<10.4f}\n")
                f.write("\n")
            
            # Summary statistics
            f.write("\nSummary Comparison:\n")
            f.write("-" * 30 + "\n")
            
            for schedule_name in self.schedules.keys():
                params = self.diffusion_params[schedule_name]
                final_signal = params['sqrt_alphas_cumprod'][-1]
                final_noise = params['sqrt_one_minus_alphas_cumprod'][-1]
                
                f.write(f"{schedule_name.title()} schedule:\n")
                f.write(f"  Final signal retention: {final_signal:.6f}\n")
                f.write(f"  Final noise strength: {final_noise:.6f}\n")
                f.write(f"  Final SNR: {final_signal/(final_noise + 1e-8):.6f}\n\n")
        
        print(f"Numerical analysis saved to {analysis_file}")
    
    def plot_schedule_curves(self, output_dir):
        """Plot the mathematical curves of the noise schedules."""
        output_dir = Path(output_dir)
        
        timesteps_array = np.arange(self.timesteps)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Beta values
        ax = axes[0, 0]
        for name, betas in self.schedules.items():
            ax.plot(timesteps_array, betas, label=name, linewidth=2)
        ax.set_title('Beta Schedules', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Alpha cumulative product (signal retention)
        ax = axes[0, 1]
        for name in self.schedules.keys():
            alphas_cumprod = self.diffusion_params[name]['alphas_cumprod']
            ax.plot(timesteps_array, alphas_cumprod, label=name, linewidth=2)
        ax.set_title('Signal Retention (α̅)', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('α̅')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Sqrt Alpha cumulative product
        ax = axes[1, 0]
        for name in self.schedules.keys():
            sqrt_alphas = self.diffusion_params[name]['sqrt_alphas_cumprod']
            ax.plot(timesteps_array, sqrt_alphas, label=name, linewidth=2)
        ax.set_title('Signal Coefficient (√α̅)', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√α̅')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Noise coefficient
        ax = axes[1, 1]
        for name in self.schedules.keys():
            sqrt_one_minus = self.diffusion_params[name]['sqrt_one_minus_alphas_cumprod']
            ax.plot(timesteps_array, sqrt_one_minus, label=name, linewidth=2)
        ax.set_title('Noise Coefficient (√(1-α̅))', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√(1-α̅)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'schedule_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Schedule curves saved to {output_dir / 'schedule_curves.png'}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize noise addition across timesteps for different schedules')
    
    parser.add_argument('--image_path', type=str, default='/user/HS402/zs00774/Downloads/data/test/00011.jpg',
                       help='Path to input image')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Total number of timesteps')
    parser.add_argument('--show_timesteps', nargs='+', type=int,
                       default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 999],
                       help='Specific timesteps to visualize')
    parser.add_argument('--output_dir', type=str, default='noise_schedule_visualization',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        return
    
    print(f"Visualizing noise schedules:")
    print(f"  Image: {args.image_path}")
    print(f"  Timesteps to show: {args.show_timesteps}")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Output: {args.output_dir}")
    
    # Create visualizer
    visualizer = NoiseScheduleVisualizer(timesteps=args.timesteps)
    
    # Generate visualizations
    visualizer.visualize_noise_progression(
        args.image_path, args.show_timesteps, args.output_dir
    )
    
    # Plot mathematical curves
    visualizer.plot_schedule_curves(args.output_dir)
    
    print(f"\nVisualization complete!")
    print(f"Check {args.output_dir} for:")
    print(f"  - noise_schedule_comparison.png: Full grid comparison")
    print(f"  - labeled_noise_comparison.png: Labeled version")
    print(f"  - timestep_XXXX_comparison.png: Per-timestep comparisons")
    print(f"  - individual_progressions/: Individual schedule progressions")
    print(f"  - schedule_curves.png: Mathematical curves")
    print(f"  - noise_analysis.txt: Numerical analysis")


if __name__ == "__main__":
    main()
