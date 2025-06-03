"""
Simple OCR Dataset and Data Generator
"""

import torch
from torch.utils.data import Dataset
from data_generator import BaseDataGenerator
from config import CHAR_TO_IDX


class CorpusDataGenerator(BaseDataGenerator):
    """Data generator for simple OCR model"""
    
    def generate_sample(self, method="corpus"):
        """
        Generate a text sample with corresponding image

        Args:
            method: Method to generate text

        Returns:
            Tuple of (text, image, language)
        """
        # For mixed language, select a language for this sample
        if self.language == "mixed":
            import random
            selected_lang = random.choice(["english", "romanian", "hungarian"])
        else:
            selected_lang = self.language

        text = self.generate_text(method=method)

        # Ensure text is not longer than max_length
        if len(text) > self.max_length:
            text = text[: self.max_length]

        image = self.render_text_to_image(text)

        return text, image, selected_lang


class OCRDataset(Dataset):
    def __init__(self, samples, transform=None, max_length=50):
        """
        OCR dataset from samples

        Args:
            samples: List of (text, image, language) tuples
            transform: PyTorch image transforms
            max_length: Maximum text length
        """
        self.samples = samples
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, image, language = self.samples[idx]

        # Apply transforms if needed
        if self.transform:
            image = self.transform(image)

        # Convert text to indices
        target = [CHAR_TO_IDX.get(char, 0) for char in text]
        # Pad sequence
        target = target + [0] * (self.max_length - len(target))
        target = torch.LongTensor(target)

        # Create target length
        target_length = min(len(text), self.max_length)
        target_length = torch.LongTensor([target_length])

        return image, target, target_length, language