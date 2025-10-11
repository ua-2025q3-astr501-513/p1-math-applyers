"""
Gradio app for COMET-SEE deployment on HuggingFace Spaces.
"""

import gradio as gr
import torch
import numpy as np
from astropy.io import fits
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import zipfile
import tempfile
from pathlib import Path
import cv2
from datetime import datetime
import base64
import io


class CometDetector:
    """Comet detection inference class."""
    
    def __init__(self, model_path='best_model.pth'):
        """Initialize the detector with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_fits_from_zip(self, zip_file):
        """Extract and load FITS images from ZIP file."""
        images = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            fits_files = sorted(Path(tmpdir).rglob('*.fts')) + sorted(Path(tmpdir).rglob('*.fits'))
            
            for fpath in fits_files[:50]:  # Limit to 50 files
                try:
                    with fits.open(fpath) as hdul:
                        img = hdul[0].data.astype(np.float32)
                        if img.shape != (1024, 1024):
                            factor = 1024 / img.shape[0]
                            img = zoom(img, factor, order=1)
                        images.append(img)
                except:
                    continue
        
        return np.array(images)
    
    def create_difference_images(self, images):
        """Create difference images and maximum projection."""
        diff_images = []
        for i in range(len(images) - 1):
            diff = images[i+1] - images[i]
            diff_images.append(diff)
        
        max_proj = np.max(np.abs(np.array(diff_images)), axis=0)
        return max_proj
    
    def classify_image(self, max_proj):
        """Classify the maximum projection image."""
        # Normalize
        img = (max_proj - max_proj.min()) / (max_proj.max() - max_proj.min() + 1e-8)
        
        # Resize
        img = cv2.resize(img, (512, 512))
        
        # Convert to RGB
        img_rgb = np.stack([img, img, img], axis=0)
        img_tensor = torch.FloatTensor(img_rgb).unsqueeze(0).to(self.device)
        
        # Apply transforms
        img_tensor = self.transform(img_tensor)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        return pred_class, confidence
    
    def generate_visualization(self, images, max_proj, pred_class, confidence):
        """Create visualization figure."""
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0a0e27')
        
        # Original frame
        axes[0].imshow(images[0], cmap='gray')
        axes[0].set_title('Original Frame', fontsize=12, color='#00d9ff')
        axes[0].axis('off')
        
        # Maximum projection
        axes[1].imshow(max_proj, cmap='hot')
        if pred_class == 1:
            axes[1].set_title('🌟 COMET DETECTED', fontsize=14, color='#00ff88', weight='bold')
        else:
            axes[1].set_title('Background', fontsize=12, color='#ffb366')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze(self, zip_file, progress=gr.Progress()):
        """Main analysis function for Gradio."""
        if zip_file is None:
            return None, "⚠️ Please upload a ZIP file"
        
        try:
            progress(0.1, desc="Extracting ZIP file...")
            images = self.load_fits_from_zip(zip_file)
            
            if len(images) < 2:
                return None, "❌ Need at least 2 FITS images in the ZIP file"
            
            progress(0.4, desc=f"Processing {len(images)} images...")
            max_proj = self.create_difference_images(images)
            
            progress(0.7, desc="Running AI classification...")
            pred_class, confidence = self.classify_image(max_proj)
            
            progress(0.9, desc="Creating visualization...")
            fig = self.generate_visualization(images, max_proj, pred_class, confidence)
            
            progress(1.0, desc="Complete!")
            
            # Generate text summary
            if pred_class == 1:
                summary = f"""
# 🌟 COMET DETECTED!

**Confidence:** {confidence:.1%}

This sequence shows characteristic signatures of a sungrazing comet.
The bright trail in the maximum projection indicates significant motion
consistent with comet activity.

---
**Images Analyzed:** {len(images)}  
**Model Accuracy:** 97.7%  
**Analysis Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            else:
                summary = f"""
# 🌑 No Comet Detected

**Background Confidence:** {confidence:.1%}

This sequence does not show signatures consistent with comet activity.
The difference projection reveals minimal motion typical of background observations.

---
**Images Analyzed:** {len(images)}  
**Model Accuracy:** 97.7%  
**Analysis Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            return fig, summary
            
        except Exception as e:
            return None, f"❌ **Error:** {str(e)}\n\nPlease ensure your ZIP contains valid SOHO FITS files."


# Initialize detector
detector = CometDetector('best_model.pth')

# Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

.gradio-container {
    font-family: 'Orbitron', sans-serif !important;
}

h1 {
    background: linear-gradient(135deg, #00d9ff 0%, #00ff88 50%, #ff00ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em !important;
    text-align: center !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="COMET-SEE") as demo:
    
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2em;">
            <h1>🌌 COMET-SEE</h1>
            <p style="font-size: 1.2em; color: #00d9ff;">
                COmet Motion Extraction & Tracking – Statistical Exploration Engine
            </p>
            <p style="color: #888;">
                AI-Powered Detection of Sungrazing Comets in SOHO Data
            </p>
        </div>
    """)
    
    gr.Markdown("""
    ### 🛸 About This System
    
    This deep learning system automatically detects sungrazing comets in SOHO/LASCO C3
    coronagraph images using difference imaging and convolutional neural networks.
    
    **Model Performance:**
    - ✨ **97.7% Accuracy** on validation data
    - 🎯 **98% Precision** for comet detection
    - 🔍 **99% Recall** for comet detection
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📤 Upload Data
            
            **Instructions:**
            1. Download SOHO LASCO C3 FITS images
            2. Place `.fts` or `.fits` files in a folder
            3. Create a ZIP archive
            4. Upload and analyze
            
            **Recommended:** 6-hour time sequence
            """)
            
            file_input = gr.File(
                label="📁 Upload ZIP File",
                file_types=[".zip"],
                type="filepath"
            )
            
            analyze_btn = gr.Button(
                "🔬 Analyze Sequence",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Visual Analysis")
            output_text = gr.Markdown(label="Detection Report")
    
    gr.Markdown("""
    ---
    
    ### 👥 Development Team
    **Shambhavi Srivastava** • **Emily Margaret Foley** • **Mohammed Sameer Syed**
    
    ### 📚 Data Source
    Training data sourced from [NASA SOHO mission](https://soho.nascom.nasa.gov/) 
    and the [Sungrazer Project](https://sungrazer.nrl.navy.mil/).
    
    ---
    *Built with Gradio • Powered by HuggingFace • 2025*
    """)
    
    # Connect button
    analyze_btn.click(
        fn=detector.analyze,
        inputs=[file_input],
        outputs=[output_plot, output_text]
    )


if __name__ == "__main__":
    demo.launch()