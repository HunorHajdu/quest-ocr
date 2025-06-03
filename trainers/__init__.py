"""
Trainers package for OCR system
"""

from .simple_trainer import OCRTrainer
from .hierarchical_trainer import HierarchicalOCRTrainer

__all__ = ["OCRTrainer", "HierarchicalOCRTrainer"]