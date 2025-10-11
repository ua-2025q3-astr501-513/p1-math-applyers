"""
Script to create difference images from FITS sequences.

Usage:
    python 02_create_difference_images.py \
        --input data/raw \
        --output data/processed
"""

import argparse
import sys
from pathlib import Path
import glob

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import SequenceProcessor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create difference images from FITS sequences'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory with raw FITS data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for difference images'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DIFFERENCE IMAGE CREATION")
    logger.info("="*60)
    
    processor = SequenceProcessor()
    
    # Process comet sequences
    logger.info("\n[1/2] Processing comet sequences...")
    comet_input = input_dir / 'comet_images'
    comet_output = output_dir / 'comet_sequences'
    
    if comet_input.exists():
        comet_results = processor.process_multiple_sequences(
            str(comet_input),
            str(comet_output),
            pattern='SOHO-*'
        )
        logger.info(f"✅ Processed {len(comet_results)} comet sequences")
    else:
        logger.warning(f"Comet directory not found: {comet_input}")
        comet_results = {}
    
    # Process background sequences
    logger.info("\n[2/2] Processing background sequences...")
    bg_input = input_dir / 'background_images'
    bg_output = output_dir / 'background_sequences'
    
    if bg_input.exists():
        bg_results = processor.process_multiple_sequences(
            str(bg_input),
            str(bg_output),
            pattern='background_*'
        )
        logger.info(f"✅ Processed {len(bg_results)} background sequences")
    else:
        logger.warning(f"Background directory not found: {bg_input}")
        bg_results = {}
    
    # Visualization (optional)
    if args.visualize and (comet_results or bg_results):
        logger.info("\nCreating visualizations...")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize one comet example
        if comet_results:
            name = list(comet_results.keys())[0]
            result = comet_results[name]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Comet Example: {name}', fontsize=14)
            
            # Original
            axes[0].imshow(result['images'][0], cmap='gray')
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            # Difference
            idx = len(result['diff_images']) // 2
            diff_max = np.percentile(np.abs(result['diff_images'][idx]), 99)
            axes[1].imshow(result['diff_images'][idx], cmap='RdBu_r',
                          vmin=-diff_max, vmax=diff_max)
            axes[1].set_title('Difference Image')
            axes[1].axis('off')
            
            # Maximum projection
            axes[2].imshow(result['max_proj'], cmap='hot')
            axes[2].set_title('Maximum Projection')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'comet_example.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization: {viz_dir / 'comet_example.png'}")
            plt.close()
        
        # Visualize one background example
        if bg_results:
            name = list(bg_results.keys())[0]
            result = bg_results[name]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Background Example: {name}', fontsize=14)
            
            axes[0].imshow(result['images'][0], cmap='gray')
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            idx = len(result['diff_images']) // 2
            diff_max = np.percentile(np.abs(result['diff_images'][idx]), 99)
            axes[1].imshow(result['diff_images'][idx], cmap='RdBu_r',
                          vmin=-diff_max, vmax=diff_max)
            axes[1].set_title('Difference Image')
            axes[1].axis('off')
            
            axes[2].imshow(result['max_proj'], cmap='hot')
            axes[2].set_title('Maximum Projection')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'background_example.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization: {viz_dir / 'background_example.png'}")
            plt.close()
    
    # Summary
    total_diffs = sum(
        len(r['diff_images']) 
        for r in list(comet_results.values()) + list(bg_results.values())
    )
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Comet sequences: {len(comet_results)}")
    logger.info(f"Background sequences: {len(bg_results)}")
    logger.info(f"Total difference images: {total_diffs}")
    logger.info(f"\nData saved to: {output_dir}")


if __name__ == '__main__':
    main()