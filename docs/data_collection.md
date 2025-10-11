# Data Collection

This document describes the data collection process for the COMET-SEE project.

## Overview

Data collection involves three main steps:
1. **Scraping comet catalogs** from the Sungrazer Project
2. **Downloading SOHO/LASCO C3 FITS images** for comet observations
3. **Downloading background sequences** for training negative examples

## Data Sources

### Sungrazer Project

- **URL:** https://sungrazer.nrl.navy.mil/
- **Purpose:** Citizen science project for discovering sungrazing comets
- **Content:** 
  - Comet discovery announcements
  - Observation times and positions
  - Links to SOHO images

### SOHO Mission

- **Instrument:** LASCO (Large Angle and Spectrometric Coronagraph)
- **Detector:** C3 (widest field of view, 3.7 to 30 solar radii)
- **Data Access:** Via SunPy's Fido interface
- **Image Format:** FITS (Flexible Image Transport System)
- **Typical Cadence:** Multiple images per hour

## Collection Process

### Step 1: Scrape Comet Catalog

```python
from src.data_collection import CometDataCollector

collector = CometDataCollector()
years = [2005, 2006, 2007, 2008, 2009, 2010]
comet_df = collector.scrape_multiple_years(years)
```

**What it does:**
- Accesses yearly comet tables on Sungrazer website
- Extracts SOHO numbers and position file URLs
- Saves to CSV catalog

**Output:**
- `comet_catalog.csv` - All discovered comets with metadata

### Step 2: Download Position Files

```python
for idx, row in comet_df.iterrows():
    times = collector.download_position_file(
        row['position_file_url'],
        row['soho_number'],
        output_dir
    )
```

**What it does:**
- Downloads text files with comet positions over time
- Extracts observation timestamps
- Saves position files locally

**Output:**
- `SOHO-XXXX_positions.txt` - Position data for each comet
- `comets_with_times.csv` - Comet observation times

**Position File Format:**
```
# SOHO-3456 observations
2015-01-15 12:34:56  512.3  487.2  ...
2015-01-15 12:48:23  518.7  491.5  ...
...
```

### Step 3: Download Comet Images

```python
from src.data_collection import SOHOImageDownloader

downloader = SOHOImageDownloader()

for idx, row in times_df.iterrows():
    files = downloader.download_for_comet(
        soho_number=row['soho_number'],
        obs_time_str=row['obs_time'],
        base_output_dir='data/comet_images',
        window_hours=3,
        max_images=50
    )
```

**What it does:**
- Queries SOHO archive for images around observation time
- Downloads FITS files via SunPy's Fido
- Organizes into folders by comet

**Time Window:**
- Default: ±3 hours around observation
- Captures comet motion through field of view
- Typically yields 10-50 images per comet

**Output Structure:**
```
data/comet_images/
├── SOHO-3456/
│   ├── 20150115_123456_xxxxxxxxxx.fts
│   ├── 20150115_124823_xxxxxxxxxx.fts
│   └── ...
├── SOHO-3457/
│   └── ...
```

### Step 4: Download Background Sequences

```python
from src.data_collection import BackgroundDownloader

bg_downloader = BackgroundDownloader()

# Get comet dates to avoid
comet_dates = set(pd.to_datetime(times_df['obs_time']).dt.date)

bg_count = bg_downloader.download_backgrounds(
    num_sequences=50,
    comet_dates=comet_dates,
    output_dir='data/background_images',
    start_year=2005,
    end_year=2021
)
```

**What it does:**
- Generates random dates NOT coinciding with comet activity
- Downloads SOHO images for these "clean" periods
- Provides negative training examples

**Safety Checks:**
- Ensures no overlap with comet observation dates
- Checks ±1 day buffer around each comet
- Randomly samples across years for diversity

**Output Structure:**
```
data/background_images/
├── background_01/
│   ├── 20100523_060000_xxxxxxxxxx.fts
│   └── ...
├── background_02/
│   └── ...
```

## Data Quality Considerations

### Comet Selection

**Inclusion Criteria:**
- Has position file available
- Observation time is valid
- Images exist in SOHO archive

**Exclusion:**
- Comets without clear observation times
- Periods with data gaps or corrupted images

### Background Selection

**Strategy:**
- Random sampling across time range
- Avoids known comet periods
- Matches comet sequence characteristics (length, cadence)

**Validation:**
- Manual inspection of sample sequences
- Check for unexpected artifacts or features

## Dataset Statistics

Based on our collection (2005-2021):

| Category | Count |
|----------|-------|
| Total Comets Catalogued | ~5000+ |
| Comets with Position Files | ~1500 |
| Comets with Images Downloaded | 498 |
| Background Sequences | 167 |
| Total Image Sequences | 665 |
| Total FITS Images | ~25,000 |

### Why Not All Comets?

- **Archive availability:** Not all observation times have archived images
- **Image quality:** Some periods have corrupted or missing data
- **Processing limits:** Time and storage constraints
- **Balance:** Need roughly 2-3x more comets than backgrounds for class balance

## Rate Limiting and Ethics

### API Usage

- **SunPy/Fido:** No strict rate limits, but be respectful
- **Sungrazer Website:** 1 second delay between requests
- **Batch Processing:** Downloads in manageable chunks

### Best Practices

```python
import time

for item in items:
    process(item)
    time.sleep(1)  # Be nice to servers
```

### Storage Management

- **Temporary Downloads:** Consider cleaning up after processing
- **Selective Download:** Download only what you need
- **Compression:** FITS files compress well with gzip

## Troubleshooting

### Common Issues

**1. No images found for comet**
- Observation time may be outside SOHO archive range
- Try expanding time window

**2. Download timeouts**
- Network connectivity issues
- Retry with exponential backoff

**3. Corrupted FITS files**
- Some files in archive are damaged
- Skip and continue with valid files

**4. Disk space**
- Monitor storage during downloads
- Clean up failed/incomplete downloads

### Error Handling

The collection code includes robust error handling:

```python
try:
    files = download_images(...)
except Exception as e:
    logger.error(f"Failed to download: {e}")
    continue  # Skip and move to next
```

## Reproducibility

To reproduce our exact dataset:

1. Use the same year range (2005-2021)
2. Use same SOHO numbers from catalog
3. Use same time windows (±3 hours)
4. Set random seed for background selection

```python
import numpy as np
np.random.seed(42)
```

## Next Steps

After collection, proceed to:
1. **Preprocessing** - Create difference images (see `preprocessing.md`)
2. **Training** - Train the classifier (see `methodology.md`)

## References

- [SOHO Mission](https://soho.nascom.nasa.gov/)
- [Sungrazer Project](https://sungrazer.nrl.navy.mil/)
- [SunPy Documentation](https://docs.sunpy.org/)
- [FITS Format Specification](https://fits.gsfc.nasa.gov/)