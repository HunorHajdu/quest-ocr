"""
Models package for OCR system
"""

from .simple_ocr import OCRModel
from .hierarchical_ocr import HierarchicalOCRModel

__all__ = ["OCRModel", "HierarchicalOCRModel"]