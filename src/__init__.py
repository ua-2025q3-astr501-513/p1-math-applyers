"""
COMET-SEE: COmet Motion Extraction & Tracking – Statistical Exploration Engine

A machine learning system for detecting sungrazing comets in SOHO/LASCO coronagraph data.
"""

from .data_collection import (
    CometDataCollector,
    SOHOImageDownloader,
    BackgroundDownloader
)

from .preprocessing import (
    FITSLoader,
    ImageResizer,
    DifferenceImageProcessor,
    SequenceProcessor
)

from .model import (
    CometClassifier,
    DifferenceImageDataset
)

__version__ = '1.0.0'
__author__ = 'Shambhavi Srivastava, Emily Margaret Foley, Mohammed Sameer Syed'

__all__ = [
    # Data collection
    'CometDataCollector',
    'SOHOImageDownloader',
    'BackgroundDownloader',
    
    # Preprocessing
    'FITSLoader',
    'ImageResizer',
    'DifferenceImageProcessor',
    'SequenceProcessor',
    
    # Model
    'CometClassifier',
    'DifferenceImageDataset',
]