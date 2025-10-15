# COMET-SEE Deployment

This directory contains the deployment files for the COMET-SEE HuggingFace Space.

## Files

- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies for deployment
- `best_model.pth` - Trained model weights (you need to copy this)

## Local Testing

To test the app locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Copy your trained model
cp ../models/best_model.pth .

# Run the app
python app.py
```

The app will be available at http://localhost:7860

## Deploying to HuggingFace Spaces

### Method 1: Web Interface

1. Go to https://huggingface.co/new-space
2. Choose "Gradio" as the SDK
3. Upload files:
   - `app.py`
   - `requirements.txt`
   - `best_model.pth`
   - `README.md` (optional, for Space description)

### Method 2: Git Push

```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/comet-see

# Copy deployment files
cp app.py requirements.txt best_model.pth comet-see/

# Commit and push
cd comet-see
git add .
git commit -m "Deploy COMET-SEE model"
git push
```

### Method 3: Python API

```python
from huggingface_hub import HfApi

api = HfApi()

# Create Space
api.create_repo(
    repo_id="YOUR_USERNAME/comet-see",
    repo_type="space",
    space_sdk="gradio"
)

# Upload files
api.upload_folder(
    folder_path="deployment",
    repo_id="YOUR_USERNAME/comet-see",
    repo_type="space"
)
```

## Configuration

The Space will use:
- **SDK:** Gradio 4.x
- **Python:** 3.10+
- **Hardware:** CPU (free tier) or GPU (upgrade for faster inference)

## Model File

**Important:** The model file (`best_model.pth`) is not included in the repository due to its size.

You must copy it from your training output:

```bash
cp ../models/best_model.pth deployment/
```

## Customization

### Changing the Theme

Edit `app.py` and modify:

```python
demo = gr.Blocks(theme=gr.themes.Soft())  # Try: Base, Default, Glass, Monochrome
```

### Adding Examples

Add example ZIP files for users to try:

```python
gr.Examples(
    examples=[
        ["examples/comet_example.zip"],
        ["examples/background_example.zip"]
    ],
    inputs=file_input
)
```

### Modifying the Interface

The Gradio interface is defined in `app.py`. Key sections:
- Header: HTML at the top
- Input: `file_input` and `analyze_btn`
- Output: `output_plot` and `output_text`
- Styling: `custom_css` variable

## Monitoring

Once deployed, monitor your Space at:
- **URL:** https://huggingface.co/spaces/YOUR_USERNAME/comet-see
- **Logs:** Click "Logs" tab in Space dashboard
- **Analytics:** View usage statistics in Space settings

## Troubleshooting

### Build Fails

- Check `requirements.txt` for version conflicts
- Ensure `best_model.pth` is uploaded
- Check logs for specific error messages

### Out of Memory

- Reduce `max_images` in `load_fits_from_zip`
- Consider upgrading to GPU hardware
- Add explicit memory cleanup:

```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Slow Inference

- Upgrade to GPU hardware (Settings → Hardware)
- Reduce image size in preprocessing
- Use model quantization for faster CPU inference

## Support

For issues:
1. Check HuggingFace Spaces documentation
2. Review Space logs
3. Open an issue on the main repository