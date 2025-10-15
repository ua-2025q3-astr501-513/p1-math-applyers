"""
Preprocessing module for COMET-SEE project.

This module handles:
- Loading FITS images
- Creating difference images
- Maximum projection computation
- Image normalization and resizing
"""

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FITSLoader:
    """Handles loading and preprocessing of FITS images."""
    
    @staticmethod
    def load_fits_image(filepath: str) -> Optional[np.ndarray]:
        """
        Load a FITS file and return image data.
        
        Args:
            filepath: Path to FITS file
            
        Returns:
            Image data as numpy array, or None if loading fails
        """
        try:
            with fits.open(filepath) as hdul:
                data = None
                
                # Find the first HDU with data
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data.astype(float)
                        break
                
                if data is None:
                    logger.warning(f"No data in {Path(filepath).name}")
                    return None
                
                # Replace NaN values with median
                data = np.nan_to_num(data, nan=np.nanmedian(data))
                return data
                
        except Exception as e:
            logger.error(f"Error loading {Path(filepath).name}: {e}")
            return None
    
    @staticmethod
    def load_sequence(image_folder: str) -> List[np.ndarray]:
        """
        Load all FITS images from a folder.
        
        Args:
            image_folder: Path to folder containing FITS files
            
        Returns:
            List of image arrays
        """
        fits_files = sorted(glob.glob(str(Path(image_folder) / '*.fts')))
        
        images = []
        for fpath in fits_files:
            img = FITSLoader.load_fits_image(fpath)
            if img is not None:
                images.append(img)
        
        logger.info(f"Loaded {len(images)} images from {image_folder}")
        return images


class ImageResizer:
    """Handles image resizing operations."""
    
    @staticmethod
    def resize_image(
        img: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target shape using bilinear interpolation.
        
        Args:
            img: Input image
            target_shape: Target (height, width)
            
        Returns:
            Resized image
        """
        if img.shape == target_shape:
            return img
        
        zoom_factors = (
            target_shape[0] / img.shape[0],
            target_shape[1] / img.shape[1]
        )
        return zoom(img, zoom_factors, order=1)
    
    @staticmethod
    def normalize_sizes(images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize all images to the same size (largest in the list).
        
        Args:
            images: List of images with potentially different sizes
            
        Returns:
            List of images normalized to same size
        """
        if not images:
            return images
        
        # Find the largest size
        shapes = [img.shape for img in images]
        target_shape = max(shapes, key=lambda x: x[0] * x[1])
        
        # Resize all images
        normalized = []
        for img in images:
            if img.shape != target_shape:
                img = ImageResizer.resize_image(img, target_shape)
            normalized.append(img)
        
        logger.info(f"Normalized {len(images)} images to {target_shape}")
        return normalized


class DifferenceImageProcessor:
    """Creates difference images and maximum projections."""
    
    @staticmethod
    def create_difference_images(images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create frame-to-frame difference images.
        
        Args:
            images: List of sequential images
            
        Returns:
            List of difference images (length = len(images) - 1)
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images to create differences")
        
        diff_images = []
        for i in range(len(images) - 1):
            diff = images[i+1] - images[i]
            diff_images.append(diff)
        
        logger.info(f"Created {len(diff_images)} difference images")
        return diff_images
    
    @staticmethod
    def create_max_projection(diff_images: List[np.ndarray]) -> np.ndarray:
        """
        Create maximum absolute difference projection.
        
        This highlights the maximum change at each pixel across all frames,
        making moving objects (like comets) highly visible.
        
        Args:
            diff_images: List of difference images
            
        Returns:
            Maximum projection image
        """
        if not diff_images:
            raise ValueError("No difference images provided")
        
        abs_diffs = np.array([np.abs(d) for d in diff_images])
        max_proj = np.max(abs_diffs, axis=0)
        
        logger.info(f"Created maximum projection from {len(diff_images)} frames")
        return max_proj
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            img: Input image
            
        Returns:
            Normalized image
        """
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min < 1e-8:
            logger.warning("Image has zero variance")
            return np.zeros_like(img)
        
        return (img - img_min) / (img_max - img_min)


class SequenceProcessor:
    """Complete pipeline for processing an image sequence."""
    
    def __init__(self):
        self.loader = FITSLoader()
        self.resizer = ImageResizer()
        self.diff_processor = DifferenceImageProcessor()
    
    def process_sequence(
        self,
        image_folder: str,
        output_folder: Optional[str] = None,
        sequence_name: Optional[str] = None,
        save_intermediates: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Complete processing pipeline for one image sequence.
        
        Args:
            image_folder: Path to folder with FITS images
            output_folder: Path to save outputs (optional)
            sequence_name: Name for this sequence (optional)
            save_intermediates: Whether to save intermediate results
            
        Returns:
            Tuple of (original_images, difference_images, max_projection)
        """
        logger.info(f"Processing sequence: {image_folder}")
        
        # Load images
        images = self.loader.load_sequence(image_folder)
        
        if len(images) < 2:
            logger.error(f"Only {len(images)} valid images - need at least 2")
            return None, None, None
        
        # Normalize sizes
        images = self.resizer.normalize_sizes(images)
        
        # Create difference images
        diff_images = self.diff_processor.create_difference_images(images)
        
        # Create maximum projection
        max_proj = self.diff_processor.create_max_projection(diff_images)
        
        # Save if requested
        if output_folder and save_intermediates:
            self._save_results(
                output_folder, 
                sequence_name or Path(image_folder).name,
                diff_images,
                max_proj
            )
        
        return images, diff_images, max_proj
    
    def _save_results(
        self,
        output_folder: str,
        sequence_name: str,
        diff_images: List[np.ndarray],
        max_proj: np.ndarray
    ) -> None:
        """Save processing results to disk."""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save difference images
        for i, diff in enumerate(diff_images):
            filename = output_path / f"{sequence_name}_diff_{i:03d}.npy"
            np.save(filename, diff)
        
        # Save maximum projection
        max_proj_file = output_path / f"{sequence_name}_max_projection.npy"
        np.save(max_proj_file, max_proj)
        
        logger.info(f"Saved results to {output_folder}")
    
    def process_multiple_sequences(
        self,
        input_base_dir: str,
        output_base_dir: str,
        pattern: str = '*'
    ) -> dict:
        """
        Process multiple image sequences.
        
        Args:
            input_base_dir: Base directory containing sequence folders
            output_base_dir: Base directory for outputs
            pattern: Pattern to match sequence folders
            
        Returns:
            Dictionary mapping sequence names to results
        """
        sequence_folders = sorted(glob.glob(str(Path(input_base_dir) / pattern)))
        
        logger.info(f"Found {len(sequence_folders)} sequences to process")
        
        results = {}
        
        for folder in sequence_folders:
            sequence_name = Path(folder).name
            output_folder = Path(output_base_dir) / sequence_name
            
            images, diff_images, max_proj = self.process_sequence(
                folder,
                str(output_folder),
                sequence_name
            )
            
            if images is not None:
                results[sequence_name] = {
                    'images': images,
                    'diff_images': diff_images,
                    'max_proj': max_proj,
                    'output_folder': str(output_folder)
                }
        
        logger.info(f"Successfully processed {len(results)} sequences")
        return results