# Data Directory

This directory contains all data for the COMET-SEE project.

## Structure

```
data/
├── raw/                          # Raw downloaded data
│   ├── comet_images/            # SOHO images for comet observations
│   │   └── SOHO-XXXX/           # One folder per comet
│   ├── comet_positions/         # Comet position files and catalogs
│   │   ├── comet_catalog.csv
│   │   ├── comets_with_times.csv
│   │   └── SOHO-XXXX_positions.txt
│   └── background_images/       # Background sequences (no comets)
│       └── background_XX/       # One folder per sequence
│
└── processed/                    # Processed difference images
    ├── comet_sequences/         # Difference images for comets
    │   └── SOHO-XXXX/
    │       ├── SOHO-XXXX_diff_000.npy
    │       ├── SOHO-XXXX_diff_001.npy
    │       └── SOHO-XXXX_max_projection.npy
    ├── background_sequences/    # Difference images for backgrounds
    │   └── background_XX/
    └── visualizations/          # Sample visualizations (optional)
```

## Data Sources

### SOHO/LASCO C3 Images

- **Source:** NASA SOHO mission
- **Instrument:** Large Angle and Spectrometric Coronagraph (LASCO)
- **Detector:** C3 (widest field of view)
- **Format:** FITS (Flexible Image Transport System)
- **Typical Size:** 1024×1024 pixels
- **Cadence:** Multiple images per hour

### Comet Catalogs

- **Source:** Sungrazer Project (NRL)
- **URL:** https://sungrazer.nrl.navy.mil/
- **Content:** Discovery information, observation times, position files
- **Format:** HTML tables + text files

## File Formats

### FITS Images (.fts, .fits)

Standard astronomical image format containing:
- Image data (2D array)
- Header with metadata (observation time, instrument settings, etc.)

Load with:
```python
from astropy.io import fits
with fits.open('image.fts') as hdul:
    data = hdul[0].data
```

### NumPy Arrays (.npy)

Processed difference images and maximum projections:
- Faster to load than FITS
- Smaller file size
- Contains only image data (no metadata)

Load with:
```python
import numpy as np
data = np.load('max_projection.npy')
```

### CSV Files (.csv)

Catalogs and metadata:
- `comet_catalog.csv`: All scraped comets
- `comets_with_times.csv`: Comets with observation times

## Data Collection

To download data, run:

```bash
python scripts/01_download_data.py --years 2005 2010 2015 --max-comets 50
```

This will:
1. Scrape comet catalogs for specified years
2. Download position files
3. Download SOHO/LASCO C3 images
4. Download background sequences

## Processing

To create difference images:

```bash
python scripts/02_create_difference_images.py
```

This will:
1. Load FITS image sequences
2. Compute frame-to-frame differences
3. Create maximum projections
4. Save as NumPy arrays

## Storage Requirements

Approximate sizes:
- Raw FITS images: ~5-10 MB per image
- Full comet sequence (50 images): ~250-500 MB
- Processed difference images: ~10-20 MB per sequence
- Maximum projections: ~4 MB each

For 500 comets + 200 backgrounds:
- **Raw data:** ~175-350 GB
- **Processed data:** ~7-14 GB

## .gitignore

The following are excluded from version control:
- `*.fts`, `*.fits` - Raw FITS images
- `*.npy` - Processed NumPy arrays
- `*.csv` - Generated catalogs (except examples)

Only code and documentation are version controlled, not the data itself.

## Data Availability

Due to the large size, data is not included in the repository.

To reproduce the dataset:
1. Run the download scripts
2. Or download preprocessed data from: [link to data repository]

## Citation

If using this data, please cite:
- NASA SOHO mission
- Sungrazer Project (NRL)
- This project (see main README)