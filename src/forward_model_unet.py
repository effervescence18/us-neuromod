#!/usr/bin/env python3
"""
Forward Model: 3D U-Net for Transcranial Focused Ultrasound Pressure Prediction
================================================================================

Predicts acoustic pressure fields from CT skull images + transducer placement.

Input:  2-channel 3D volume (CT scan + transducer element mask)
Output: 1-channel 3D volume (predicted pressure field)

Dataset: TFUScapes (https://huggingface.co/datasets/vinkle-srivastav/TFUScapes)
    - 85 skulls, ~20 transducer placements each = 1686 total samples
    - Each sample: 256³ CT volume + transducer coordinates + k-Wave pressure field

Usage examples:
    # Full dataset training on institutional GPU (recommended: A100 40GB+)
    python forward_model_unet.py --data_dir /path/to/tfuscapes/data --mode full

    # Development mode: 10 skulls only, for prototyping
    python forward_model_unet.py --data_dir /path/to/tfuscapes/data --mode dev

    # Resume from checkpoint
    python forward_model_unet.py --data_dir /path/to/tfuscapes/data --resume checkpoints/best_model.pt

    # Download dataset first (only need to run once)
    python forward_model_unet.py --download_only --data_dir /path/to/tfuscapes/data

    # Custom hyperparameters
    python forward_model_unet.py --data_dir /path/to/data --base_features 32 --epochs 100 --lr 3e-4

Authors: Arthur & collaborator
Repository: github.com/effervescence18/us-neuromod
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# 1. MODEL ARCHITECTURE
# ============================================================================
# The U-Net is built from three reusable blocks, stacked into an
# encoder-bottleneck-decoder structure with skip connections.
# ============================================================================


class ConvBlock(nn.Module):
    """
    Two consecutive 3D convolutions, each followed by BatchNorm and ReLU.

    This is the atomic building block — every encoder and decoder level
    uses one of these. Two convolutions stacked gives an effective 5³
    receptive field (each 3³ conv sees the other's output).

    Args:
        in_channels:  number of input feature maps
        out_channels: number of output feature maps
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """
    ConvBlock + MaxPool downsampling.

    Returns BOTH the full-resolution features (for skip connections)
    and the downsampled output (for the next encoder level).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv_block(x)       # process at current resolution
        downsampled = self.pool(features)    # halve each spatial dim
        return features, downsampled


class DecoderBlock(nn.Module):
    """
    Upsample + concatenate skip connection + ConvBlock.

    The transposed convolution doubles spatial dims. The skip connection
    from the encoder provides fine spatial detail that the bottleneck
    representation is too coarse to preserve.

    Args:
        in_channels:   channels from the level below (or bottleneck)
        skip_channels: channels from the corresponding encoder level
        out_channels:  channels after processing the concatenated features
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.upsample(x)
        x = torch.cat([x, skip_features], dim=1)  # concatenate along channels
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric image-to-image prediction.

    Architecture (base_features=16, num_levels=4):

        Encoder:  2→16 → 16→32 → 32→64 → 64→128
        Bottleneck: 128→256
        Decoder:  256→128 → 128→64 → 64→32 → 32→16
        Output:   16→1

    Channel counts double at each level (standard U-Net convention).
    Skip connections link each encoder level to its decoder counterpart.

    Args:
        in_channels:   input channels (2 = CT + transducer mask)
        out_channels:  output channels (1 = pressure field)
        base_features: channels at the first encoder level
        num_levels:    number of encoder/decoder levels
    """
    def __init__(self, in_channels=2, out_channels=1, base_features=16, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Channel progression: [16, 32, 64, 128] for base=16, levels=4
        channel_list = [base_features * (2 ** i) for i in range(num_levels)]
        bottleneck_channels = channel_list[-1] * 2  # 256

        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for channels in channel_list:
            self.encoders.append(EncoderBlock(prev_channels, channels))
            prev_channels = channels

        # Bottleneck (lowest resolution, highest channel count)
        self.bottleneck = ConvBlock(channel_list[-1], bottleneck_channels)

        # Decoder path (mirrors encoder in reverse)
        self.decoders = nn.ModuleList()
        prev_channels = bottleneck_channels
        for channels in reversed(channel_list):
            self.decoders.append(DecoderBlock(prev_channels, channels, channels))
            prev_channels = channels

        # Final 1x1x1 conv: collapse features to single-channel prediction
        self.final_conv = nn.Conv3d(channel_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, 2, D, H, W)

        # Encoder — save skip connections
        skip_connections = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — use skip connections in reverse order
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        return self.final_conv(x)


# ============================================================================
# 2. LOSS FUNCTION
# ============================================================================


class WeightedMSELoss(nn.Module):
    """
    MSE loss weighted by target pressure magnitude.

    Without weighting, the model can predict zeros everywhere and achieve
    low loss (>99% of the volume IS near-zero). The focal spot is a tiny
    region that contributes negligibly to unweighted MSE.

    Weighting: w(voxel) = 1 + alpha * (target / max_target)
        - Background voxels: weight ≈ 1.0 (still contribute)
        - Focal spot peak:   weight ≈ 1 + alpha (emphasized)

    Args:
        alpha: emphasis strength (1.0 = focal spot gets 2x weight)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction, target):
        squared_error = (prediction - target) ** 2

        # Normalize target to [0, 1] per sample for weighting
        target_max = target.flatten(1).max(dim=1).values
        target_max = target_max.view(-1, 1, 1, 1, 1)
        weights = 1.0 + self.alpha * (target / (target_max + 1e-8))

        return (weights * squared_error).mean()


# ============================================================================
# 3. DATASET
# ============================================================================


def make_transducer_mask(tr_coords, volume_shape):
    """
    Convert transducer element coordinates → binary 3D mask.

    Args:
        tr_coords: (N, 3) array of voxel positions for each transducer element
        volume_shape: spatial dimensions of the output mask (D, H, W)

    Returns:
        Binary mask with 1.0 at transducer positions, 0.0 elsewhere
    """
    mask = np.zeros(volume_shape, dtype=np.float32)
    coords = tr_coords.astype(int)
    for dim in range(3):
        coords[:, dim] = np.clip(coords[:, dim], 0, volume_shape[dim] - 1)
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return mask


class TFUScapesDataset(Dataset):
    """
    PyTorch Dataset for TFUScapes ultrasound simulation data.

    Each sample is a .npz file containing:
        - ct:        (256, 256, 256) CT scan of the skull
        - tr_coords: (N, 3) transducer element positions in voxel space
        - pmap:      (256, 256, 256) k-Wave simulated pressure field

    Preprocessing pipeline:
        1. Normalize CT by dividing by ct_max (maps HU → ~[0, 1])
        2. Log-transform pressure: log(1 + p) to compress dynamic range
        3. Downsample both volumes to target_size³ (memory constraint)
        4. Build transducer mask at downsampled resolution

    Args:
        data_dir:    path to the directory containing skull_*/experiment_*.npz
        file_list:   list of relative paths like "skull_001/experiment_015.npz"
        target_size: spatial resolution to downsample to (128 = 128³)
        ct_max:      normalization constant for CT values
    """
    def __init__(self, data_dir, file_list, target_size=128, ct_max=2000.0):
        self.data_dir = data_dir
        self.file_list = file_list
        self.ct_max = ct_max
        self.scale_factor = target_size / 256

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.file_list[index])
        data = np.load(filepath)

        ct = data['ct']
        tr_coords = data['tr_coords']
        pmap = data['pmap']

        # Normalize CT to ~[0, 1]
        ct = ct / self.ct_max

        # Log-transform pressure to compress dynamic range
        pmap = np.log1p(pmap)

        # Downsample to target resolution
        ct = zoom(ct, self.scale_factor, order=1)
        pmap = zoom(pmap, self.scale_factor, order=1)

        # Build transducer mask at downsampled resolution
        tr_mask = make_transducer_mask(
            (tr_coords * self.scale_factor).astype(int),
            ct.shape
        )

        # Stack CT + mask as 2-channel input
        input_volume = np.stack([ct, tr_mask], axis=0)     # (2, D, H, W)
        input_tensor = torch.from_numpy(input_volume).float()
        target_tensor = torch.from_numpy(pmap).float().unsqueeze(0)  # (1, D, H, W)

        return input_tensor, target_tensor


# ============================================================================
# 4. DATA DOWNLOAD (only needed once)
# ============================================================================


def download_dataset(data_dir, mode="full"):
    """
    Download TFUScapes data from Hugging Face.

    The HF dataset object is just a manifest (filenames as text).
    Actual .npz volumes are downloaded individually via hf_hub_download.

    Args:
        data_dir: where to store downloaded files
        mode:     "full" for all 85 skulls, "dev" for first 10 skulls
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print("Loading file manifest from Hugging Face...")
    ds = load_dataset("vinkle-srivastav/TFUScapes")

    train_files = [item['text'] for item in ds['train']]
    val_files = [item['text'] for item in ds['validation']]

    if mode == "dev":
        train_skulls = sorted(set(f.split('/')[0] for f in train_files))
        dev_skulls = set(train_skulls[:10])
        train_files = [f for f in train_files if f.split('/')[0] in dev_skulls]
        print(f"Dev mode: using {len(dev_skulls)} skulls, "
              f"{len(train_files)} train + {len(val_files)} val files")
    else:
        print(f"Full mode: {len(train_files)} train + {len(val_files)} val files")

    all_files = train_files + val_files
    print(f"Downloading {len(all_files)} files to {data_dir}...")

    for i, filename in enumerate(all_files):
        output_path = os.path.join(data_dir, 'data', filename)
        if os.path.exists(output_path):
            continue

        hf_hub_download(
            repo_id="vinkle-srivastav/TFUScapes",
            filename=f"data/{filename}",
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False,
        )

        if (i + 1) % 25 == 0:
            print(f"  Downloaded {i + 1}/{len(all_files)}")

    print("Download complete!")
    return train_files, val_files


def get_file_lists(data_dir, mode="full"):
    """
    Get train/val file lists without downloading (assumes data exists).
    """
    from datasets import load_dataset
    ds = load_dataset("vinkle-srivastav/TFUScapes")

    train_files = [item['text'] for item in ds['train']]
    val_files = [item['text'] for item in ds['validation']]

    if mode == "dev":
        train_skulls = sorted(set(f.split('/')[0] for f in train_files))
        dev_skulls = set(train_skulls[:10])
        train_files = [f for f in train_files if f.split('/')[0] in dev_skulls]

    # Verify at least some files exist
    sample_path = os.path.join(data_dir, 'data', train_files[0])
    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            f"Data not found at {sample_path}. "
            f"Run with --download_only first, or check --data_dir path."
        )

    return train_files, val_files


# ============================================================================
# 5. TRAINING ENGINE
# ============================================================================


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, log_interval=10):
    """Run one training epoch with mixed-precision."""
    model.train()
    total_loss = 0
    num_batches = len(loader)

    for batch_idx, (input_volume, target) in enumerate(loader):
        input_volume = input_volume.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            prediction = model(input_volume)
            loss = criterion(prediction, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logging.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation pass (no gradients, no weight updates)."""
    model.eval()
    total_loss = 0

    for input_volume, target in loader:
        input_volume = input_volume.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            prediction = model(input_volume)
            loss = criterion(prediction, target)

        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                    checkpoint_dir, filename="checkpoint.pt"):
    """Save full training state for resumability."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)
    logging.info(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None, device='cuda'):
    """Load checkpoint. Returns the epoch number to resume from."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    logging.info(
        f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1} | "
        f"Train loss: {checkpoint['train_loss']:.4f} | "
        f"Val loss: {checkpoint['val_loss']:.4f}"
    )
    return checkpoint['epoch'] + 1  # return next epoch to train


# ============================================================================
# 6. VISUALIZATION (optional, for evaluation)
# ============================================================================


def visualize_predictions(model, dataset, device, output_dir, num_samples=4, tag=""):
    """
    Generate comparison plots: CT | Ground Truth | Prediction | Error.
    Saves to output_dir as PNG files.
    """
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Evenly spaced samples across the validation set
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        input_volume, target = dataset[idx]
        input_volume = input_volume.unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                prediction = model(input_volume)

        pred = prediction[0, 0].cpu().numpy()
        gt = target[0].numpy()
        ct = input_volume[0, 0].cpu().numpy()

        # Find the axial slice with peak pressure
        focal_slice = gt.max(axis=(1, 2)).argmax()

        vmin = min(gt[focal_slice].min(), pred[focal_slice].min())
        vmax = max(gt[focal_slice].max(), pred[focal_slice].max())

        axes[row, 0].imshow(ct[focal_slice], cmap='gray')
        axes[row, 0].set_title(f'CT (sample {idx})')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt[focal_slice], cmap='hot', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred[focal_slice], cmap='hot', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title('Prediction')
        axes[row, 2].axis('off')

        diff = np.abs(gt[focal_slice] - pred[focal_slice])
        axes[row, 3].imshow(diff, cmap='coolwarm')
        axes[row, 3].set_title('Absolute Error')
        axes[row, 3].axis('off')

    plt.suptitle(f'Validation Predictions {tag}', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'predictions{tag}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"  Visualization saved: {save_path}")


# ============================================================================
# 7. MAIN: Argument parsing + training orchestration
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for tFUS pressure field prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data first (run once)
  python forward_model_unet.py --download_only --data_dir /data/tfuscapes

  # Train on full dataset
  python forward_model_unet.py --data_dir /data/tfuscapes --mode full --epochs 100

  # Dev mode (10 skulls, for prototyping)
  python forward_model_unet.py --data_dir /data/tfuscapes --mode dev --epochs 20

  # Resume training
  python forward_model_unet.py --data_dir /data/tfuscapes --resume runs/run_001/checkpoints/best_model.pt

  # Scale up model
  python forward_model_unet.py --data_dir /data/tfuscapes --base_features 32 --batch_size 2
        """
    )

    # --- Required ---
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for TFUScapes data')

    # --- Mode ---
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'dev'],
                        help='full = all 85 skulls, dev = 10 skulls (default: full)')
    parser.add_argument('--download_only', action='store_true',
                        help='Download data and exit (no training)')

    # --- Model ---
    parser.add_argument('--base_features', type=int, default=16,
                        help='Channels at first encoder level (default: 16, try 32 for bigger model)')
    parser.add_argument('--num_levels', type=int, default=4,
                        help='Number of encoder/decoder levels (default: 4)')
    parser.add_argument('--target_size', type=int, default=128,
                        help='Downsample volumes to this resolution (default: 128)')

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1; 256³ volumes are memory-heavy)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for Adam (default: 1e-4)')
    parser.add_argument('--weight_alpha', type=float, default=1.0,
                        help='Spatial weighting emphasis in loss (default: 1.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4)')

    # --- Checkpointing ---
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume from')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for this run (default: auto-generated)')

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Handle download-only mode ---
    if args.download_only:
        download_dataset(args.data_dir, mode=args.mode)
        return

    # --- Create output directory for this run ---
    if args.output_dir:
        run_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"{timestamp}_{args.mode}_bf{args.base_features}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    viz_dir = os.path.join(run_dir, "visualizations")

    # --- Logging: both console and file ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(run_dir, 'training.log'))
        ]
    )

    # Save the exact config for reproducibility
    config = vars(args)
    config['run_dir'] = run_dir
    config['timestamp'] = datetime.now().isoformat()
    config['pytorch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['gpu_name'] = torch.cuda.get_device_name(0)
        config['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_mem / 1e9

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logging.info("=" * 60)
    logging.info("Forward Model: 3D U-Net for tFUS Pressure Prediction")
    logging.info("=" * 60)
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Config: {json.dumps(config, indent=2)}")

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    if device.type == 'cpu':
        logging.warning("Running on CPU — training will be very slow!")

    # --- Data ---
    logging.info("Loading file lists...")
    actual_data_dir = os.path.join(args.data_dir, 'data')
    train_files, val_files = get_file_lists(args.data_dir, mode=args.mode)
    logging.info(f"Train: {len(train_files)} samples | Val: {len(val_files)} samples")

    train_dataset = TFUScapesDataset(actual_data_dir, train_files,
                                      target_size=args.target_size)
    val_dataset = TFUScapesDataset(actual_data_dir, val_files,
                                    target_size=args.target_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,      # faster CPU→GPU transfer
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # --- Model ---
    model = UNet3D(
        in_channels=2,
        out_channels=1,
        base_features=args.base_features,
        num_levels=args.num_levels,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")

    # --- Optimizer, loss, scaler ---
    criterion = WeightedMSELoss(alpha=args.weight_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    # --- Resume from checkpoint if specified ---
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, device)
        logging.info(f"Resuming from epoch {start_epoch}")

    # --- Training loop ---
    logging.info(f"Starting training: epochs {start_epoch + 1} to {start_epoch + args.epochs}")
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': []}

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)

        logging.info(
            f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Time: {epoch_time:.0f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scaler, epoch, train_loss, val_loss,
                checkpoint_dir, filename="best_model.pt"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, train_loss, val_loss,
                checkpoint_dir, filename=f"checkpoint_epoch_{epoch}.pt"
            )

        # Periodic visualization
        if epoch % args.save_every == 0:
            visualize_predictions(
                model, val_dataset, device, viz_dir,
                num_samples=4, tag=f"_epoch{epoch}"
            )

    # --- Final save ---
    save_checkpoint(
        model, optimizer, scaler, epoch, train_loss, val_loss,
        checkpoint_dir, filename="final_model.pt"
    )

    # Save training history
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logging.info("=" * 60)
    logging.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logging.info(f"Results in: {run_dir}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
