import argparse
import torch
from pathlib import Path
import sys

# --- ADD THIS BLOCK ---
# Add the path to the cloned PTQ4DM repository.
# This assumes the 'PTQ4DM' folder is in the same directory as your 'idkyet_ffid' project folder.
# Adjust the path if you cloned it elsewhere.
try:
    # This path works if you run the script from inside the 'idkyet_ffid' directory
    #ptq4dm_path = Path(__file__).parent.parent / 'PTQ4DM'
    #sys.path.append(str(ptq4dm_path))
    from PTQ4DM.quant import quantize_model
except ImportError:
    # Fallback if the above path is wrong, you can set it manually
    print("Could not find PTQ4DM library. Please update the path in quantize_model.py")
    sys.exit(1)
# --- END OF BLOCK ---

# Imports from your existing codebase
from unet import UNetModel, DiffusionInpaintingModel
from data.dataset import create_inference_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Apply Post-Training Quantization to a fine-tuned inpainting model.')
    
    # Model paths
    parser.add_argument('--finetuned-checkpoint', type=str, required=True,
                        help='Path to the fully fine-tuned model checkpoint (e.g., best_model.pt).')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the final quantized model state dictionary.')

    # Data paths for calibration
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing validation images for calibration.')
    parser.add_argument('--mask-dir', type=str, required=True,
                        help='Directory containing mask subdirectories (train/val/test).')
    
    # Quantization settings
    parser.add_argument('--cali-samples', type=int, default=128,
                        help='Number of samples to use for calibration.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for the calibration dataloader.')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size for the model and data.')

    return parser.parse_args()

def main():
    """Main function to load, quantize, and save the model."""
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Your Fully Fine-tuned Model ---
    print("Loading the fine-tuned inpainting model...")
    
    # Step 1: Create the base UNet architecture. 
    # This must match the architecture of your saved checkpoint.
    base_model = UNetModel(
        image_size=args.img_size,
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
    
    # Step 2: Wrap it in your DiffusionInpaintingModel to get the 9-channel input layer
    model = DiffusionInpaintingModel(base_model, in_channels=9).to(device)

    # Step 3: Load the saved state dictionary from your fine-tuning process
    checkpoint = torch.load(args.finetuned_checkpoint, map_location=device)
    
    # Handle checkpoints that might be saved in different formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded fine-tuned model.")

    # --- 2. Create the Calibration Dataloader ---
    print(f"Loading {args.cali_samples} samples for calibration...")
    
    # Use your existing dataloader function to get calibration data
    cali_loader = create_inference_dataloader(
        test_dir=args.data_dir,
        mask_dir=args.mask_dir,
        split='val', # Use the validation set for calibration
        batch_size=args.batch_size,
        img_size=args.img_size,
        random_samples=args.cali_samples # Limit to the number of calibration samples
    )
    print("Calibration dataloader created.")

    # --- 3. Run Post-Training Quantization ---
    print("Starting quantization process with PTQ4DM...")
    
    # The PTQ4DM library requires a forward pass wrapper for calibration.
    # This adapts the data from the loader to your model's expected input format.
    def forward_wrapper(batch):
        # The dataloader provides a dictionary with all necessary components
        noisy_image = batch['image'].to(device) # For calibration, we can treat the clean image as the noisy input
        masked_image = batch['masked_image'].to(device)
        mask = batch['mask'].to(device)
        
        # A dummy timestep is sufficient for calibration
        t = torch.randint(0, 1000, (noisy_image.shape[0],), device=device).long()

        # Your model expects a 9-channel input for inpainting
        inpainting_input = torch.cat([noisy_image, masked_image, mask.repeat(1, 3, 1, 1)], dim=1)
        return model(inpainting_input, t)

    # Pass the model and dataloader to the PTQ4DM function
    quantized_model = quantize_model(
        model=model,
        forward_wrapper=forward_wrapper,
        dataloader=cali_loader,
        device=device
    )
    print("Quantization complete.")

    # --- 4. Save the Quantized Model ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
