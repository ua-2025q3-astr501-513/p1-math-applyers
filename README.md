# COMET-SEE 🌟

**COmet Motion Extraction & Tracking – Statistical Exploration Engine**

An AI-powered system for automatically detecting sungrazing comets in NASA SOHO/LASCO coronagraph images using deep learning and difference imaging techniques.

[![Model Performance](https://img.shields.io/badge/Accuracy-97.7%25-brightgreen)]()
[![Precision](https://img.shields.io/badge/Precision-98%25-blue)]()
[![Recall](https://img.shields.io/badge/Recall-99%25-blue)]()
[![HuggingFace](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/MohammedSameerSyed/soho-comet-detector)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Team](#team)
- [Citation](#citation)

## 🔭 Overview

COMET-SEE automatically detects and classifies comet activity in SOHO (Solar and Heliospheric Observatory) LASCO C3 coronagraph images. The system uses difference imaging to highlight moving objects, then applies a convolutional neural network to classify sequences as containing comet activity or background noise.

**Key Statistics:**
- **Dataset:** 498 comet sequences + 167 background sequences
- **Time Range:** 2005-2021
- **Source:** SOHO/LASCO C3 coronagraph
- **Model:** EfficientNet-B0
- **Validation Accuracy:** 97.7%

## ✨ Features

- **Automated Data Collection:** Scrapes comet catalogs from Sungrazer Project
- **Smart Image Download:** Fetches SOHO/LASCO C3 FITS images for comet observation times
- **Difference Imaging:** Creates maximum projection images to highlight motion
- **Deep Learning Classification:** EfficientNet-B0 model for binary classification
- **Web Deployment:** Interactive Gradio interface on HuggingFace Spaces

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Google Drive account (for Colab notebooks)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/comet-see.git
cd comet-see

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

If using Google Colab (recommended):

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/yourusername/comet-see.git
%cd comet-see
!pip install -r requirements.txt
```

## 📊 Usage

### 1. Data Collection

Download SOHO comet data and background sequences:

```bash
python scripts/01_download_data.py --years 2005-2021 --output data/raw
```

### 2. Preprocessing

Create difference images from FITS sequences:

```bash
python scripts/02_create_difference_images.py \
    --input data/raw \
    --output data/processed
```

### 3. Model Training

Train the EfficientNet-B0 classifier:

```bash
python scripts/03_train_model.py \
    --data data/processed \
    --epochs 30 \
    --batch-size 16 \
    --output models/
```

### 4. Deployment (Optional)

Deploy to HuggingFace Spaces:

```bash
cd deployment
# Update credentials in config
python deploy.py
```

## 🔬 Methodology

### Difference Imaging

1. **Load FITS Sequence:** Read time-series SOHO/LASCO C3 images
2. **Create Differences:** Compute frame-to-frame differences
3. **Maximum Projection:** Take absolute maximum across all differences
4. **Normalize:** Scale to [0, 1] for neural network input

### Model Architecture

- **Base Model:** EfficientNet-B0 (pretrained on ImageNet)
- **Input Size:** 512×512×3 (grayscale converted to RGB)
- **Output:** Binary classification (Background / Comet)
- **Training:**
  - Optimizer: Adam (lr=0.0001)
  - Loss: CrossEntropyLoss
  - Scheduler: ReduceLROnPlateau
  - Data Augmentation: Random flips, rotations

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.7% |
| Precision (Comet) | 98% |
| Recall (Comet) | 99% |
| F1-Score | 98.5% |

### Confusion Matrix

```
                Predicted
              Background  Comet
Actual  Bg        152       2
        Comet       1      98
```

## 👥 Team

- **Shambhavi Srivastava** - Data Collection & Alignment Analysis
- **Emily Margaret Foley** - Data Collection, Preprocessing & Trajectory
- **Mohammed Sameer Syed** - Model Training & Deployment

## 📚 Citation

If you use this work, please cite:

```bibtex
@software{comet_see_2025,
  title={COMET-SEE: Automated Comet Detection in SOHO Data},
  author={Srivastava, Shambhavi and Foley, Margaret Emily and Syed, Mohammed Sameer},
  year={2025},
  url={https://github.com/yourusername/comet-see}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA SOHO Mission for providing open-access coronagraph data
- [Sungrazer Project](https://sungrazer.nrl.navy.mil/) for comet catalogs
- SunPy project for solar physics tools
- HuggingFace for hosting the demo

## 📞 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [mohammedsameer@arizona.edu]

---

**Live Demo:** [https://huggingface.co/spaces/MohammedSameerSyed/soho-comet-detector](https://huggingface.co/spaces/MohammedSameerSyed/soho-comet-detector)