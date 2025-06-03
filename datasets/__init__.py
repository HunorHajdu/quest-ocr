"""
Datasets package for OCR system
"""

from .simple_dataset import OCRDataset, CorpusDataGenerator
from .hierarchical_dataset import HierarchicalOCRDataset, HierarchicalDataGenerator

__all__ = ["OCRDataset", "CorpusDataGenerator", "HierarchicalOCRDataset", "HierarchicalDataGenerator"]