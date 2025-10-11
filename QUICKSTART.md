# COMET-SEE Quickstart Guide

Get started with COMET-SEE in 10 minutes! This guide will walk you through the complete pipeline.

## Prerequisites

- **Python:** 3.8 or higher
- **Storage:** 50+ GB for dataset
- **GPU:** Recommended for training (not required for inference)
- **Internet:** For downloading SOHO data

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/comet-see.git
cd comet-see
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
python -c "import sunpy; import torch; import timm; print('✅ All dependencies installed!')"
```

## Pipeline Walkthrough

### Step 1: Download Data (30-60 minutes)

Download SOHO comet data and background sequences:

```bash
python scripts/01_download_data.py \
    --output data/raw \
    --years 2015 2016 2017 \
    --max-comets 20 \
    --max-backgrounds 10
```

**What this does:**
- Scrapes comet catalog from Sungrazer Project
- Downloads SOHO/LASCO C3 FITS images
- Creates background sequences
- Saves everything to `data/raw/`

**Expected output:**
```
✅ Total comets in catalog: 156
✅ Comets with observation times: 45
✅ Comet sequences downloaded: 20
✅ Background sequences downloaded: 10
```

### Step 2: Preprocess Data (10-20 minutes)

Create difference images from FITS sequences:

```bash
python scripts/02_create_difference_images.py \
    --input data/raw \
    --output data/processed \
    --visualize
```

**What this does:**
- Loads FITS image sequences
- Creates frame-to-frame differences
- Computes maximum projections
- Saves as NumPy arrays

**Expected output:**
```
✅ Processed 20 comet sequences
✅ Processed 10 background sequences
✅ Total difference images: 450
```

**Check your results:**
```bash
ls data/processed/comet_sequences/SOHO-*/
# Should show *_diff_*.npy and *_max_projection.npy files
```

### Step 3: Train Model (1-3 hours)

Train the EfficientNet-B0 classifier:

```bash
python scripts/03_train_model.py \
    --data data/processed \
    --output models \
    --epochs 30 \
    --batch-size 16
```

**What this does:**
- Loads maximum projection images
- Splits into train/validation sets
- Trains EfficientNet-B0
- Saves best model checkpoint

**Expected output:**
```
Epoch 30/30
Train Loss: 0.0145 | Train Acc: 98.7%
Val Loss: 0.0389 | Val Acc: 97.7%
✅ Training complete! Best Val Acc: 97.7%
```

**Check your results:**
```bash
ls models/
# Should show: best_model.pth, training_curves.png, confusion_matrix.png
```

### Step 4: Evaluate Model

View the results:

```bash
# Open the confusion matrix
open models/confusion_matrix.png  # Mac
xdg-open models/confusion_matrix.png  # Linux
start models/confusion_matrix.png  # Windows

# Read the summary
cat models/results_summary.txt
```

### Step 5: Deploy (Optional)

Deploy to HuggingFace Spaces:

```bash
cd deployment

# Copy your trained model
cp ../models/best_model.pth .

# Test locally
python app.py
# Visit http://localhost:7860

# Deploy to HuggingFace (requires HF account)
# See deployment/README.md for details
```

## Usage Examples

### Example 1: Analyze a Single Sequence

```python
from src.preprocessing import SequenceProcessor
from src.model import CometClassifier

# Load sequence
processor = SequenceProcessor()
images, diffs, max_proj = processor.process_sequence(
    'data/raw/comet_images/SOHO-3456/'
)

# Classify
classifier = CometClassifier()
classifier.load('models/best_model.pth')

# Predict
prediction, confidence = classifier.classify_image(max_proj)

if prediction == 1:
    print(f"🌟 COMET DETECTED! (Confidence: {confidence:.1%})")
else:
    print(f"🌑 Background (Confidence: {confidence:.1%})")
```

### Example 2: Batch Processing

```python
import glob
from pathlib import Path

# Process all sequences
comet_folders = glob.glob('data/raw/comet_images/SOHO-*')

results = []
for folder in comet_folders:
    name = Path(folder).name
    _, _, max_proj = processor.process_sequence(folder)
    pred, conf = classifier.classify_image(max_proj)
    
    results.append({
        'name': name,
        'prediction': 'Comet' if pred == 1 else 'Background',
        'confidence': conf
    })

# Print summary
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values('confidence', ascending=False))
```

### Example 3: Create Visualization

```python
from src.utils import visualize_sequence

# Load and visualize
images, diffs, max_proj = processor.process_sequence(
    'data/raw/comet_images/SOHO-3456/'
)

visualize_sequence(
    images, diffs, max_proj,
    title='SOHO-3456 Analysis',
    save_path='output_visualization.png'
)
```

## Common Issues

### Issue 1: Out of Memory

**Symptom:** Crash during training with "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python scripts/03_train_model.py --batch-size 8
```

### Issue 2: Downloads Fail

**Symptom:** Connection timeouts or missing images

**Solution:**
- Check internet connection
- Try downloading fewer sequences at once
- Increase timeout in `src/data_collection.py`

### Issue 3: Corrupted FITS Files

**Symptom:** "Error loading FITS file" messages

**Solution:**
- This is normal! Some files in the archive are corrupted
- The code skips them automatically
- You need at least 2 valid images per sequence

### Issue 4: Import Errors

**Symptom:** `ModuleNotFoundError`

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/comet-see

# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

### Improve Performance

**Get more data:**
```bash
python scripts/01_download_data.py \
    --years 2005 2006 2007 2008 2009 2010 \
    --max-comets 500 \
    --max-backgrounds 200
```

**Train longer:**
```bash
python scripts/03_train_model.py --epochs 50
```

**Try different models:**
```python
# In src/model.py, change:
model = timm.create_model('efficientnet_b1', ...)  # Larger model
```

### Explore the Code

- **Data collection:** `src/data_collection.py`
- **Preprocessing:** `src/preprocessing.py`
- **Model:** `src/model.py`
- **Utilities:** `src/utils.py`

### Read the Documentation

- **Data Collection:** `docs/data_collection.md`
- **Methodology:** `docs/methodology.md`
- **Performance:** `docs/model_performance.md`

### Try the Demo

Visit the live demo:
🌐 **https://huggingface.co/spaces/MohammedSameerSyed/soho-comet-detector**

## Troubleshooting

### Get Help

1. **Check documentation:** Look in `docs/` folder
2. **Review examples:** See code in `scripts/`
3. **Open an issue:** GitHub Issues page
4. **Contact team:** See README for emails

### Debug Mode

Run with verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in scripts:
```bash
python scripts/01_download_data.py --verbose
```

## Project Structure Recap

```
comet-see/
├── README.md              ← Start here!
├── QUICKSTART.md         ← You are here
├── requirements.txt       ← Dependencies
├── data/                  ← Your datasets
├── src/                   ← Core library
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── scripts/              ← Run these!
│   ├── 01_download_data.py
│   ├── 02_create_difference_images.py
│   └── 03_train_model.py
├── deployment/           ← Deploy to web
│   └── app.py
├── docs/                 ← Documentation
│   ├── data_collection.md
│   ├── methodology.md
│   └── model_performance.md
└── models/              ← Your trained models
```

## Minimum Working Example

The absolute minimum to see results:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download small dataset (10 minutes)
python scripts/01_download_data.py --years 2015 --max-comets 5 --max-backgrounds 2

# 3. Preprocess (2 minutes)
python scripts/02_create_difference_images.py

# 4. Train quick model (30 minutes)
python scripts/03_train_model.py --epochs 10 --batch-size 8

# 5. See results
cat models/results_summary.txt
```

## Tips for Success

✅ **Start small:** Use 10-20 sequences for testing
✅ **Check output:** Verify each step before moving on
✅ **Monitor resources:** Keep eye on disk space and memory
✅ **Save often:** Don't lose your trained models!
✅ **Read docs:** Understand the methodology

## Success Checklist

- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Data downloaded (at least 10 sequences)
- [ ] Difference images created
- [ ] Model trained (>90% accuracy)
- [ ] Results visualized
- [ ] (Optional) Deployed to HuggingFace

## Congratulations! 🎉

You've completed the COMET-SEE pipeline! 

Now you can:
- Process more data to improve accuracy
- Deploy your model for real-time detection
- Contribute improvements to the project
- Use it for your own astronomical research

**Happy comet hunting!** 🌟

---

*For detailed information, see the full [README.md](README.md)*