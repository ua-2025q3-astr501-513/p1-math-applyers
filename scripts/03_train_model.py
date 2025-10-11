"""
Script to train the comet detection model.

Usage:
    python 03_train_model.py \
        --data data/processed \
        --output models \
        --epochs 30 \
        --batch-size 16
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.model import CometClassifier, DifferenceImageDataset
from torch.utils.data import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train comet detection model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed',
        help='Directory with processed difference images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for trained model'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=512,
        help='Image size for model input'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for validation'
    )
    
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple:
    """Load all maximum projection images."""
    logger.info("Loading data...")
    
    # Load comet images
    comet_folders = sorted(glob.glob(str(data_dir / 'comet_sequences' / '*')))
    comet_images = []
    
    for folder in comet_folders:
        max_proj_files = glob.glob(f'{folder}/*max_projection.npy')
        if max_proj_files:
            img = np.load(max_proj_files[0])
            comet_images.append(img)
    
    logger.info(f"Loaded {len(comet_images)} comet images")
    
    # Load background images
    bg_folders = sorted(glob.glob(str(data_dir / 'background_sequences' / '*')))
    bg_images = []
    
    for folder in bg_folders:
        max_proj_files = glob.glob(f'{folder}/*max_projection.npy')
        if max_proj_files:
            img = np.load(max_proj_files[0])
            bg_images.append(img)
    
    logger.info(f"Loaded {len(bg_images)} background images")
    
    # Create labels
    comet_labels = [1] * len(comet_images)
    bg_labels = [0] * len(bg_images)
    
    all_images = comet_images + bg_images
    all_labels = comet_labels + bg_labels
    
    return all_images, all_labels


def main():
    """Main execution function."""
    args = parse_args()
    
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("COMET-SEE MODEL TRAINING")
    logger.info("="*60)
    
    # Load data
    all_images, all_labels = load_data(data_dir)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        all_images,
        all_labels,
        test_size=args.test_size,
        stratify=all_labels,
        random_state=42
    )
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(X_train)} ({sum(y_train)} comets)")
    logger.info(f"  Val: {len(X_val)} ({sum(y_val)} comets)")
    
    # Create classifier
    classifier = CometClassifier()
    
    # Create datasets
    train_transform = classifier.get_transforms(augment=True)
    val_transform = classifier.get_transforms(augment=False)
    
    train_dataset = DifferenceImageDataset(
        X_train, y_train, 
        transform=train_transform,
        img_size=args.img_size
    )
    
    val_dataset = DifferenceImageDataset(
        X_val, y_val,
        transform=val_transform,
        img_size=args.img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    logger.info("\nStarting training...")
    
    # Train model
    model_path = output_dir / 'best_model.pth'
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=str(model_path)
    )
    
    # Plot training curves
    logger.info("\nCreating training plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved training curves: {output_dir / 'training_curves.png'}")
    
    # Evaluate on validation set
    logger.info("\nEvaluating model...")
    
    classifier.load(str(model_path))
    results = classifier.evaluate(val_loader)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Background', 'Comet'],
        yticklabels=['Background', 'Comet']
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")
    
    # Save results summary
    summary_file = output_dir / 'results_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("COMET-SEE MODEL TRAINING RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Final Validation Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("-"*50 + "\n")
        
        for label, metrics in results['report'].items():
            if isinstance(metrics, dict):
                f.write(f"\n{label}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
    
    logger.info(f"Saved results summary: {summary_file}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Final Validation Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Precision (Comet): {results['report']['Comet']['precision']:.2%}")
    logger.info(f"Recall (Comet): {results['report']['Comet']['recall']:.2%}")
    logger.info(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()