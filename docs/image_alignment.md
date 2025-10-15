# Image Alignment using Fourier Shift Theorem

This document describes the image alignment process used in COMET-SEE to correct for spacecraft jitter and improve image quality.

## Overview

SOHO spacecraft experiences small shifts and rotations during imaging, causing misalignment between consecutive frames. This alignment step uses **phase cross-correlation** in the Fourier domain to:

1. Detect sub-pixel shifts between images
2. Correct these shifts with minimal interpolation blur
3. Improve signal-to-noise ratio for difference imaging

## Method: Fourier Shift Theorem

### Theory

The **Fourier Shift Theorem** states that a spatial shift in the image domain corresponds to a phase shift in the Fourier domain:

```
If g(x,y) = f(x-Δx, y-Δy)
Then G(u,v) = F(u,v) · e^(-2πi(uΔx + vΔy))
```

**Advantages:**
- **Sub-pixel accuracy:** Can achieve 0.01-pixel precision
- **Minimal blur:** No interpolation artifacts
- **Fast:** FFT is O(N log N)
- **Robust:** Works even with noise

### Pipeline Steps

#### 1. Preprocessing

```python
def preprocess(image):
    # Fix NaN values
    image = np.nan_to_num(image, nan=np.nanmedian(image))
    
    # Median-center (remove bias)
    image = image - np.nanmedian(image)
    
    # Variance normalize (equalize contrast)
    image = image / np.nanstd(image)
    
    # Apodize edges (reduce FFT ringing)
    window = cosine_window(image.shape, frac=0.08)
    image = image * window
    
    return image
```

**Why preprocess?**
- **NaN handling:** Replace with median to avoid FFT issues
- **Normalization:** Makes images comparable in brightness
- **Apodization:** Reduces edge artifacts from periodic FFT assumption

#### 2. Reference Frame

Two options:

**Option A: First Frame**
- Fast, simple
- May be noisy or have artifacts

**Option B: Median of First N Frames** (Recommended)
- More robust to outliers
- Better SNR
- Slower but worth it

```python
# Median reference from first 10 frames
ref_stack = [preprocess(read_image(f)) for f in files[:10]]
reference = np.median(np.stack(ref_stack, axis=0), axis=0)
```

#### 3. Phase Cross-Correlation

For each image, estimate the shift relative to reference:

```python
shift, error, phase_diff = phase_cross_correlation(
    reference,
    image,
    upsample_factor=50,  # 0.02 pixel precision
    normalization=None
)
```

**Returns:**
- `shift`: (dy, dx) in pixels
- `error`: Registration error (lower is better)
- `phase_diff`: Phase difference (unused here)

**How it works:**
1. FFT both images
2. Compute cross-power spectrum
3. Find peak in inverse FFT
4. Upsample around peak for sub-pixel accuracy

#### 4. Apply Shift

Apply the detected shift using Fourier domain:

```python
# Transform to Fourier domain
F = np.fft.fftn(image_raw)

# Apply shift (note: negative to move image onto reference)
F_shifted = fourier_shift(F, shift=(-dy, -dx))

# Transform back
aligned = np.fft.ifftn(F_shifted).real
```

**Note:** We apply the shift to the **raw** (unprocessed) image to preserve original pixel values.

## Usage

### Basic Usage

```bash
python scripts/05_align_images_fft.py \
    --input data/raw/comet_images/SOHO-3456 \
    --output data/aligned/SOHO-3456
```

### Advanced Options

```bash
python scripts/05_align_images_fft.py \
    --input data/raw/comet_images/SOHO-3456 \
    --output data/aligned/SOHO-3456 \
    --upsample 100 \              # Higher precision (slower)
    --median-ref \                # Use median reference
    --median-n 15 \               # Use 15 frames for median
    --apodize \                   # Apply edge apodization
    --save-stacks \               # Create mean/median stacks
    --save-format same            # Match input format
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Input directory with FITS/images |
| `--output` | Required | Output directory for aligned images |
| `--upsample` | 50 | Sub-pixel precision (50 = 0.02 px) |
| `--median-ref` | False | Use median of first N frames |
| `--median-n` | 10 | Number of frames for median |
| `--apodize` | False | Apply cosine edge window |
| `--save-stacks` | False | Save mean/median stacks |
| `--save-format` | same | Output format (same/npy) |
| `--recursive` | False | Search subdirectories |

## Output Files

### Aligned Images

- **FITS input:** `*_aligned.fits`
- **Image input:** `*_aligned.png` or `*_aligned.npy`

### Shift Log (`shifts.csv`)

```csv
file,dy,dx,err,success
/path/to/image1.fits,-0.234,0.567,0.001,True
/path/to/image2.fits,-0.189,0.543,0.002,True
...
```

**Columns:**
- `file`: Original file path
- `dy`: Vertical shift in pixels (negative = move up)
- `dx`: Horizontal shift in pixels (negative = move left)
- `err`: Registration error (lower is better)
- `success`: Whether alignment succeeded

### Optional Stacks

If `--save-stacks` is used:
- `stack_mean.fits/png`: Mean of all aligned frames (high SNR)
- `stack_median.fits/png`: Median of aligned frames (robust to outliers)

## Quality Control

### 1. Check Shifts

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load shifts
df = pd.read_csv('data/aligned/SOHO-3456/shifts.csv')

# Plot shift trajectory
plt.figure(figsize=(10, 5))
plt.plot(df['dx'], df['dy'], 'o-')
plt.xlabel('dx (pixels)')
plt.ylabel('dy (pixels)')
plt.title('Spacecraft Drift')
plt.grid(True)
plt.show()

# Check registration errors
print(f"Median error: {df['err'].median():.4f}")
print(f"90th percentile: {df['err'].quantile(0.9):.4f}")
```

**Good alignment:**
- Errors < 0.01 (excellent)
- Errors 0.01-0.05 (good)
- Errors > 0.1 (check these frames)

### 2. Visual Inspection

Create overlay of original vs aligned:

```python
import numpy as np
import matplotlib.pyplot as plt

orig = read_image('original.fits')
aligned = read_image('original_aligned.fits')

# Normalize for display
def norm(x):
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / (hi - lo), 0, 1)

# Red/Green overlay
overlay = np.stack([norm(orig), norm(aligned), np.zeros_like(orig)], axis=-1)

plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title('Red=Original, Green=Aligned (Yellow=Good)')
plt.axis('off')
plt.show()
```

**Good alignment:** Features appear yellow (red + green)
**Misalignment:** Red and green fringes around features

### 3. Filter Bad Frames

Use MAD (Median Absolute Deviation) to filter outliers:

```python
def mad_filter(arr, k=6.0):
    """Robust outlier detection."""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return np.abs(arr - med) <= k * 1.4826 * mad

# Load shifts
df = pd.read_csv('shifts.csv')

# Filter
good = (df['success'] & 
        mad_filter(df['dy']) & 
        mad_filter(df['dx']) & 
        mad_filter(df['err']))

print(f"Good frames: {good.sum()} / {len(df)}")

# Use only good frames for stacking
good_files = df.loc[good, 'file'].values
```

## Integration with COMET-SEE Pipeline

### Before Difference Imaging

**Recommended workflow:**

```bash
# 1. Download data
python scripts/01_download_data.py --output data/raw

# 2. Align images (NEW STEP)
python scripts/05_align_images_fft.py \
    --input data/raw/comet_images/SOHO-3456 \
    --output data/aligned/SOHO-3456 \
    --median-ref --apodize --save-stacks

# 3. Create difference images (use aligned data)
python scripts/02_create_difference_images.py \
    --input data/aligned/SOHO-3456 \
    --output data/processed/SOHO-3456

# 4. Continue with training...
```

### Benefits for Comet Detection

**Without alignment:**
- Noisy difference images
- False detections from jitter
- Reduced comet signal

**With alignment:**
- Clean difference images
- Comet trails more visible
- Better model performance

**Expected improvements:**
- 10-20% better SNR in difference images
- 5-10% fewer false