"""
Utility functions for COMET-SEE project.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def visualize_sequence(
    images: List[np.ndarray],
    diff_images: List[np.ndarray],
    max_proj: np.ndarray,
    title: str = "Sequence Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive visualization of image sequence processing.
    
    Args:
        images: Original images
        diff_images: Difference images
        max_proj: Maximum projection
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Original frame
    ax1 = plt.subplot(2, 3, 1)
    vmin, vmax = np.percentile(images[0], [1, 99])
    ax1.imshow(images[0], cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('First Frame', fontsize=12)
    ax1.axis('off')
    
    # Last frame
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(images[-1], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title('Last Frame', fontsize=12)
    ax2.axis('off')
    
    # Difference image (middle)
    ax3 = plt.subplot(2, 3, 3)
    idx = len(diff_images) // 2
    diff_max = np.percentile(np.abs(diff_images[idx]), 99)
    im = ax3.imshow(diff_images[idx], cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    ax3.set_title('Difference (Moving Objects)', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # Maximum projection
    ax4 = plt.subplot(2, 1, 2)
    im2 = ax4.imshow(max_proj, cmap='hot')
    ax4.set_title('Maximum Difference Projection - Comet Track Visible Here',
                 fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, fraction=0.046, label='Change Intensity')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Background', 'Comet'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix array
        class_names: Names of classes
        save_path: Path to save figure (optional)
    """
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
        'f1': f1_score(labels, predictions, average='binary'),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }
    
    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary with classification metrics
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*60 + "\n")


def create_gif_from_sequence(
    images: List[np.ndarray],
    output_path: str,
    duration: int = 200,
    normalize: bool = True
) -> None:
    """
    Create animated GIF from image sequence.
    
    Args:
        images: List of images
        output_path: Path to save GIF
        duration: Duration per frame in milliseconds
        normalize: Whether to normalize images
    """
    from PIL import Image
    
    pil_images = []
    
    for img in images:
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img_8bit = (img * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_8bit))
    
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )
    
    logger.info(f"Saved GIF to {output_path}")


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in MB
    """
    import os
    return os.path.getsize(filepath) / (1024 * 1024)


def estimate_storage_requirements(
    num_comets: int,
    num_backgrounds: int,
    images_per_sequence: int = 50,
    mb_per_image: float = 5
) -> dict:
    """
    Estimate storage requirements for dataset.
    
    Args:
        num_comets: Number of comet sequences
        num_backgrounds: Number of background sequences
        images_per_sequence: Average images per sequence
        mb_per_image: Average MB per FITS image
        
    Returns:
        Dictionary with storage estimates
    """
    total_sequences = num_comets + num_backgrounds
    total_images = total_sequences * images_per_sequence
    
    raw_storage_mb = total_images * mb_per_image
    processed_storage_mb = total_sequences * 4  # ~4 MB per max projection
    
    return {
        'total_sequences': total_sequences,
        'total_images': total_images,
        'raw_storage_gb': raw_storage_mb / 1024,
        'processed_storage_gb': processed_storage_mb / 1024,
        'total_storage_gb': (raw_storage_mb + processed_storage_mb) / 1024
    }


def print_storage_estimate(estimate: dict) -> None:
    """Print formatted storage estimate."""
    print("\n" + "="*60)
    print("STORAGE REQUIREMENTS ESTIMATE")
    print("="*60)
    print(f"Total Sequences: {estimate['total_sequences']}")
    print(f"Total Images: {estimate['total_images']}")
    print(f"\nStorage Needs:")
    print(f"  Raw FITS Images:       {estimate['raw_storage_gb']:.1f} GB")
    print(f"  Processed Arrays:      {estimate['processed_storage_gb']:.1f} GB")
    print(f"  Total:                 {estimate['total_storage_gb']:.1f} GB")
    print("="*60 + "\n")