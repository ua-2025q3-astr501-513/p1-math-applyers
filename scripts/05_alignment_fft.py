"""
Image alignment using Fourier Shift Theorem for COMET-SEE project.

This script aligns SOHO/LASCO images using phase cross-correlation
to correct for spacecraft jitter and improve image quality before
creating difference images.

Usage:
    python 05_align_images_fft.py \
        --input data/raw/comet_images/SOHO-3456 \
        --output data/aligned/SOHO-3456 \
        --upsample 50
"""

import os
import sys
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from astropy.io import fits
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# I/O UTILITIES
# ============================================================================

def is_fits(path):
    """Check if file is FITS format."""
    p = path.lower()
    return p.endswith((".fits", ".fit", ".fts", ".fz", ".fits.gz", ".fit.gz"))


def list_image_files(folder, recursive=False):
    """List all image files in folder."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", 
            "*.fits", "*.fit", "*.fts", "*.fz")
    files = []
    
    for ext in exts:
        if recursive:
            pattern = os.path.join(folder, "**", ext)
            files += glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(folder, ext)
            files += glob.glob(pattern)
    
    # Exclude GIFs
    files = [f for f in files if not f.lower().endswith(".gif")]
    files.sort()
    
    return files


def read_image(path):
    """
    Read image file (FITS or regular format).
    Returns 2D float32 array.
    """
    if is_fits(path):
        with fits.open(path, memmap=False) as hdul:
            data = None
            # Find first HDU with 2D image data
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    arr = np.asarray(hdu.data)
                    if arr.ndim >= 2:
                        data = arr
                        break
            
            if data is None:
                raise ValueError(f"No image data in {os.path.basename(path)}")
            
            # Collapse >2D to first plane
            if data.ndim > 2:
                data = data[0]
            
            return data.astype(np.float32)
    else:
        import imageio.v3 as iio
        arr = iio.imread(path)
        arr = np.asarray(arr)
        
        # Convert RGB to grayscale
        if arr.ndim == 3:
            arr = 0.2989*arr[..., 0] + 0.5870*arr[..., 1] + 0.1140*arr[..., 2]
        
        # Ensure 2D
        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Image not 2D: {path}")
        
        return arr.astype(np.float32)


def save_image(path_out, arr, like_path=None, save_format="same"):
    """Save aligned image in appropriate format."""
    if is_fits(like_path):
        hdu = fits.PrimaryHDU(np.array(arr, dtype=np.float32))
        hdu.writeto(path_out, overwrite=True)
    else:
        if save_format == "npy":
            np.save(path_out, arr.astype(np.float32))
        else:
            import imageio.v3 as iio
            # Robust scaling to 0-255
            lo, hi = np.nanpercentile(arr, (1, 99))
            if hi <= lo:
                lo, hi = arr.min(), arr.max()
            scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
            iio.imwrite(path_out, (scaled * 255).astype(np.uint8))


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def nan_to_num(arr):
    """Replace NaNs with median."""
    return np.nan_to_num(arr, nan=np.nanmedian(arr), posinf=None, neginf=None)


def normalize_image(arr):
    """Median-center and variance-scale."""
    arr = arr - np.nanmedian(arr)
    std = np.nanstd(arr)
    if std > 0:
        arr = arr / std
    return arr


def cosine_window(shape, frac=0.1):
    """
    Create 2D Tukey-like apodization window.
    Softly tapers 'frac' of pixels at each edge.
    """
    ny, nx = shape
    wy = np.ones(ny, dtype=np.float32)
    wx = np.ones(nx, dtype=np.float32)
    
    eyy = int(frac * ny)
    exx = int(frac * nx)
    
    if eyy > 0:
        ramp = (1 - np.cos(np.linspace(0, np.pi, 2 * eyy))) / 2
        wy[:eyy] = ramp[:eyy]
        wy[-eyy:] = ramp[eyy:]
    
    if exx > 0:
        ramp = (1 - np.cos(np.linspace(0, np.pi, 2 * exx))) / 2
        wx[:exx] = ramp[:exx]
        wx[-exx:] = ramp[exx:]
    
    return np.outer(wy, wx).astype(np.float32)


def preprocess(arr, window=None):
    """Preprocess image for alignment: fix NaNs, normalize, apodize."""
    arr = nan_to_num(arr.astype(np.float32))
    arr = normalize_image(arr)
    if window is not None:
        arr = arr * window
    return arr


# ============================================================================
# SHAPE FILTERING
# ============================================================================

def fast_shape(path):
    """Get image shape from metadata without loading pixels."""
    if is_fits(path):
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            for hdu in hdul:
                hdr = hdu.header
                if hdr.get("NAXIS", 0) >= 2 and "NAXIS1" in hdr and "NAXIS2" in hdr:
                    return int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        raise ValueError("No 2D image HDU")
    else:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            return (h, w)


def filter_by_dominant_shape(files, max_workers=8):
    """
    Filter files to only include those with the dominant shape.
    This prevents mixing different image sizes.
    """
    logger.info("Checking image shapes...")
    
    def shape_or_none(f):
        try:
            return (f, fast_shape(f))
        except Exception:
            return (f, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        shape_info = list(executor.map(shape_or_none, files))
    
    # Find dominant shape
    shapes = [sh for _, sh in shape_info if sh is not None]
    if not shapes:
        raise ValueError("No valid images found")
    
    shape_counts = Counter(shapes)
    target_shape = shape_counts.most_common(1)[0][0]
    
    logger.info(f"Dominant shape: {target_shape}")
    logger.info(f"Shape distribution: {dict(shape_counts)}")
    
    # Keep only matching files
    filtered = [f for f, sh in shape_info if sh == target_shape]
    
    logger.info(f"Filtered to {len(filtered)} images with shape {target_shape}")
    
    return filtered, target_shape


# ============================================================================
# ALIGNMENT CORE
# ============================================================================

def create_reference(files, use_median=True, n_frames=10, window=None):
    """
    Create reference image for alignment.
    
    Args:
        files: List of image file paths
        use_median: If True, use median of first n_frames; else use first frame
        n_frames: Number of frames for median reference
        window: Optional apodization window
    
    Returns:
        ref: Preprocessed reference
        ref_raw: Raw reference (unprocessed)
    """
    if use_median:
        n = min(n_frames, len(files))
        logger.info(f"Creating median reference from {n} frames...")
        
        stack = []
        stack_raw = []
        
        for f in files[:n]:
            img = read_image(f)
            stack.append(preprocess(img, window=window))
            stack_raw.append(img)
        
        ref = np.median(np.stack(stack, axis=0), axis=0)
        ref_raw = np.median(np.stack(stack_raw, axis=0), axis=0)
    else:
        logger.info("Using first frame as reference")
        ref_raw = read_image(files[0])
        ref = preprocess(ref_raw, window=window)
    
    return ref, ref_raw


def align_images(files, output_dir, reference, window=None, 
                 upsample=50, save_format="same"):
    """
    Align all images to reference using phase cross-correlation.
    
    Args:
        files: List of image paths
        output_dir: Output directory
        reference: Preprocessed reference image
        window: Apodization window
        upsample: Subpixel precision factor
        save_format: Output format ("same", "npy")
    
    Returns:
        shifts: List of shift dictionaries
        aligned_paths: List of output paths
    """
    shifts = []
    aligned_paths = []
    
    for f in tqdm(files, desc="Aligning images"):
        try:
            # Read and preprocess
            img_raw = read_image(f).astype(np.float32)
            img = preprocess(img_raw, window=window)
            
            # Estimate shift
            shift_rc, error, _ = phase_cross_correlation(
                reference, img,
                upsample_factor=upsample,
                normalization=None
            )
            
            dy, dx = float(shift_rc[0]), float(shift_rc[1])
            
            # Apply shift in Fourier domain
            F = np.fft.fftn(nan_to_num(img_raw))
            F_shifted = fourier_shift(F, shift=(-dy, -dx))
            aligned = np.fft.ifftn(F_shifted).real.astype(np.float32)
            
            # Save
            base_name = os.path.basename(f)
            name, ext = os.path.splitext(base_name)
            
            if is_fits(f):
                out_path = os.path.join(output_dir, name + "_aligned.fits")
            else:
                if save_format == "npy":
                    out_path = os.path.join(output_dir, name + "_aligned.npy")
                else:
                    out_path = os.path.join(output_dir, name + "_aligned.png")
            
            save_image(out_path, aligned, like_path=f, save_format=save_format)
            aligned_paths.append(out_path)
            
            shifts.append({
                "file": f,
                "dy": dy,
                "dx": dx,
                "err": float(error),
                "success": True
            })
            
        except Exception as e:
            logger.warning(f"Failed to align {f}: {e}")
            shifts.append({
                "file": f,
                "dy": np.nan,
                "dx": np.nan,
                "err": np.nan,
                "success": False
            })
    
    return shifts, aligned_paths


def save_shifts(shifts, output_path):
    """Save alignment shifts to CSV."""
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, 
            fieldnames=["file", "dy", "dx", "err", "success"]
        )
        writer.writeheader()
        for row in shifts:
            writer.writerow(row)


def create_stacks(aligned_paths, output_dir, files):
    """Create mean and median stacks from aligned images."""
    logger.info("Creating stacks...")
    
    arrs = [read_image(p).astype(np.float32) for p in aligned_paths]
    stack = np.stack(arrs, axis=0)
    
    mean_stack = np.nanmean(stack, axis=0)
    median_stack = np.nanmedian(stack, axis=0)
    
    # Determine output format
    out_ext = ".fits" if is_fits(files[0]) else ".png"
    
    out_mean = os.path.join(output_dir, f"stack_mean{out_ext}")
    out_median = os.path.join(output_dir, f"stack_median{out_ext}")
    
    save_image(out_mean, mean_stack, like_path=files[0])
    save_image(out_median, median_stack, like_path=files[0])
    
    logger.info(f"Saved stacks: {out_mean}, {out_median}")
    
    return mean_stack, median_stack


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Align SOHO images using Fourier shift theorem"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for aligned images'
    )
    
    parser.add_argument(
        '--upsample',
        type=int,
        default=50,
        help='Subpixel precision (1=pixel, 50=0.02px)'
    )
    
    parser.add_argument(
        '--median-ref',
        action='store_true',
        help='Use median of first N frames as reference'
    )
    
    parser.add_argument(
        '--median-n',
        type=int,
        default=10,
        help='Number of frames for median reference'
    )
    
    parser.add_argument(
        '--apodize',
        action='store_true',
        help='Apply cosine edge apodization'
    )
    
    parser.add_argument(
        '--save-stacks',
        action='store_true',
        help='Also save mean and median stacks'
    )
    
    parser.add_argument(
        '--save-format',
        choices=['same', 'npy'],
        default='same',
        help='Output format for non-FITS images'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for images recursively'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("="*60)
    logger.info("IMAGE ALIGNMENT (FOURIER SHIFT THEOREM)")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Upsample factor: {args.upsample}")
    
    # List files
    files = list_image_files(args.input, recursive=args.recursive)
    
    if not files:
        raise ValueError(f"No images found in {args.input}")
    
    logger.info(f"Found {len(files)} candidate images")
    
    # Filter by shape
    files, shape = filter_by_dominant_shape(files)
    
    # Read first image to set up window
    first_img = read_image(files[0])
    window = cosine_window(first_img.shape, frac=0.08) if args.apodize else None
    
    # Create reference
    reference, ref_raw = create_reference(
        files,
        use_median=args.median_ref,
        n_frames=args.median_n,
        window=window
    )
    
    # Align images
    shifts, aligned_paths = align_images(
        files,
        args.output,
        reference,
        window=window,
        upsample=args.upsample,
        save_format=args.save_format
    )
    
    # Save shifts
    csv_path = os.path.join(args.output, "shifts.csv")
    save_shifts(shifts, csv_path)
    logger.info(f"Saved shifts to {csv_path}")
    
    # Create stacks if requested
    if args.save_stacks and aligned_paths:
        mean_stack, median_stack = create_stacks(
            aligned_paths,
            args.output,
            files
        )
    
    # Summary
    success_count = sum(1 for s in shifts if s['success'])
    logger.info("="*60)
    logger.info("ALIGNMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Successfully aligned: {success_count}/{len(files)}")
    logger.info(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()