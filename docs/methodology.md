# Methodology

This document describes the complete methodology for detecting sungrazing comets in SOHO data.

## Overview

COMET-SEE uses a two-stage approach:

1. **Difference Imaging** - Highlight moving objects by subtracting sequential frames
2. **Deep Learning Classification** - Use CNN to classify sequences as comet/background

## Stage 1: Difference Imaging

### Motivation

**Problem:** Comets are faint moving objects against a bright, complex background (solar corona, stars, planets, cosmic rays).

**Solution:** Difference imaging removes static background, highlighting only moving/changing objects.

### Process

#### 1. Load Image Sequence

```python
from src.preprocessing import FITSLoader

loader = FITSLoader()
images = loader.load_sequence('path/to/comet/images/')
```

- Load all FITS files in chronological order
- Handle corrupted files gracefully
- Replace NaN values with median

#### 2. Normalize Image Sizes

```python
from src.preprocessing import ImageResizer

images = ImageResizer.normalize_sizes(images)
```

**Why?** SOHO images occasionally have different sizes due to:
- Instrument mode changes
- Data transmission errors
- Calibration updates

**Solution:** Resize all to largest dimensions using bilinear interpolation

#### 3. Create Difference Images

```python
diff_images = []
for i in range(len(images) - 1):
    diff = images[i+1] - images[i]
    diff_images.append(diff)
```

**Key Concept:**
- Static objects (stars, corona) → near zero difference
- Moving objects (comets) → positive or negative values
- Cosmic rays → isolated bright pixels

#### 4. Maximum Projection

```python
max_proj = np.max(np.abs(diff_images), axis=0)
```

**Key Concept:**
- Takes absolute value of all differences
- Finds maximum change at each pixel
- Creates single image showing comet track

**Result:** Comet appears as bright trail where it moved across frames

### Mathematical Formulation

Given sequence of images I₁, I₂, ..., Iₙ:

**Difference images:**
```
D_i = I_{i+1} - I_i  for i = 1 to n-1
```

**Maximum projection:**
```
M(x,y) = max{|D_1(x,y)|, |D_2(x,y)|, ..., |D_{n-1}(x,y)|}
```

Where (x,y) are pixel coordinates.

### Example

**Original Frame:**
```
[100, 100, 100, 100]  ← Background
[100, 100, 100, 100]
[100, 100, 100, 100]
[100, 105, 100, 100]  ← Comet here
```

**Next Frame:**
```
[100, 100, 100, 100]
[100, 100, 100, 100]
[100, 100, 105, 100]  ← Comet moved right
[100, 100, 100, 100]
```

**Difference:**
```
[  0,   0,   0,   0]
[  0,   0,   0,   0]
[  0,   0, +5,   0]  ← New position
[  0,  -5,   0,   0]  ← Old position
```

**After Max Projection:**
Trail becomes visible as sequence of differences accumulates.

## Stage 2: Deep Learning Classification

### Model Architecture

**Base Model:** EfficientNet-B0

**Why EfficientNet?**
- **Efficient:** Good accuracy with fewer parameters
- **Scalable:** Family of models (B0 to B7)
- **Pretrained:** Transfer learning from ImageNet
- **Lightweight:** Fast inference for deployment

**Modifications:**
- Input: 512×512×3 (grayscale converted to RGB)
- Output: 2 classes (Background, Comet)
- Pretrained on ImageNet, finetuned on our data

### Network Details

```
Input (512×512×3)
    ↓
EfficientNet-B0 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Fully Connected (1280 → 2)
    ↓
Softmax
    ↓
Output [P(background), P(comet)]
```

**Parameters:** ~5.3M total

### Training Configuration

**Optimizer:** Adam
- Learning rate: 0.0001
- Weight decay: default

**Loss Function:** CrossEntropyLoss
- Standard for multi-class classification
- Handles class imbalance well with our dataset

**Scheduler:** ReduceLROnPlateau
- Monitors validation accuracy
- Reduces LR by 0.5 when plateau detected
- Patience: 5 epochs

**Data Augmentation:**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why these augmentations?**
- **Flips:** Comets can appear from any direction
- **Rotation:** SOHO images have no fixed orientation
- **Normalization:** ImageNet statistics for transfer learning

### Training Process

#### 1. Data Preparation

```python
# Load maximum projections
comet_images = load_comet_max_projections()
background_images = load_background_max_projections()

# Create labels
comet_labels = [1] * len(comet_images)
bg_labels = [0] * len(background_images)

# Combine and split
X = comet_images + background_images
y = comet_labels + bg_labels

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### 2. Training Loop

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_acc = evaluate(model, val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_acc:
        save_model(model)
```

#### 3. Early Stopping

Model automatically saves when validation accuracy improves, preventing overfitting.

## Dataset Split

**Training Set:** 80% of data
- Used for learning model parameters
- Data augmentation applied

**Validation Set:** 20% of data
- Used for hyperparameter tuning
- Model selection (best checkpoint)
- No data augmentation

**Stratification:** Maintains class balance in both splits

## Class Balance

**Dataset Composition:**
- Comets: 498 sequences (~75%)
- Backgrounds: 167 sequences (~25%)

**Ratio:** 3:1 (comets:backgrounds)

**Why this balance?**
- Reflects real-world discovery rate
- Comets are the positive class of interest
- 3:1 ratio works well without reweighting

## Inference Pipeline

```python
# 1. Load ZIP of FITS images
images = load_fits_from_zip(zip_file)

# 2. Create difference images
diff_images = create_differences(images)

# 3. Maximum projection
max_proj = create_max_projection(diff_images)

# 4. Preprocess for model
tensor = preprocess(max_proj)

# 5. Classify
prediction, confidence = model.predict(tensor)

# 6. Return result
return "Comet" if prediction == 1 else "Background"
```

## Key Design Decisions

### Why Not Object Detection?

**Considered:** YOLO, Faster R-CNN for detecting comets directly

**Chose Full-Image Classification Instead:**
- Comets are extended objects, not compact
- Trail length varies significantly
- Full sequence context more informative
- Simpler pipeline, easier deployment

### Why Maximum Projection?

**Alternative:** Feed all difference images to 3D CNN or LSTM

**Maximum Projection Advantages:**
- Single 2D image easier to handle
- Highlights strongest signal (peak brightness)
- Reduces computational requirements
- Works well empirically

### Why EfficientNet?

**Alternatives:** ResNet, VGG, ViT

**EfficientNet Advantages:**
- Best accuracy/parameter trade-off
- Fast inference (important for deployment)
- Strong transfer learning from ImageNet
- Well-supported (timm library)

## Validation Strategy

### Metrics

**Primary:** Validation Accuracy
- Overall correctness
- Easy to interpret

**Secondary:** Precision and Recall
- Precision: How many predicted comets are real?
- Recall: How many real comets are detected?

**Confusion Matrix:**
```
                Predicted
              Background  Comet
Actual  Bg        TN       FP
        Comet     FN       TP
```

### Cross-Validation

**Note:** We use a single train/val split (80/20) rather than k-fold CV because:
- Dataset is large enough (665 sequences)
- Training is computationally expensive
- Results are stable with this split

## Limitations and Considerations

### Temporal Resolution

- Time windows fixed at ±3 hours
- May miss very fast or very slow comets
- Could be optimized per comet

### Image Quality

- Some FITS files are corrupted
- Cosmic ray hits can confuse difference imaging
- Brightness variations affect normalization

### Generalization

- Trained only on SOHO/LASCO C3 data
- May not transfer to other instruments
- Limited to sungrazing comets

### False Positives

**Sources:**
- Planets passing through FOV
- Bright stars
- Data artifacts
- Cosmic ray showers

**Mitigation:**
- Manual filtering of training data
- Careful background selection
- High confidence threshold in deployment

## Future Improvements

### Potential Enhancements

1. **Multi-Scale Analysis:** Process at multiple resolutions
2. **Temporal Modeling:** Use LSTM to model motion explicitly
3. **Attention Mechanisms:** Focus on relevant regions
4. **Ensemble Methods:** Combine multiple models
5. **Active Learning:** Human-in-the-loop for edge cases

### Data Augmentation

Additional augmentations to try:
- Gaussian noise injection
- Brightness/contrast variation
- Elastic deformation
- Cutout/mixup

## Reproducibility

### Random Seeds

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

### Environment

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Hyperparameters

All hyperparameters documented in:
- `scripts/03_train_model.py`
- Default values provided
- Command-line arguments for tuning

## References

**Methods:**
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs
- Alard & Lupton (1998). A Method for Optimal Image Subtraction

**SOHO Science:**
- Biesecker et al. (2002). Sungrazing Comets Discovered with SOHO
- Knight et al. (2010). SOHO's Cometary Legacy

## Next Steps

After understanding the methodology:
1. Review model performance metrics (`model_performance.md`)
2. Try training on your own data
3. Deploy to production (`deployment/README.md`)