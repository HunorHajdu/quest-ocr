"""
Hierarchical OCR Dataset and Data Generator
"""

import torch
from torch.utils.data import Dataset
from data_generator import BaseDataGenerator
from config import CHAR_TO_IDX, LANGUAGE_TO_IDX, NGRAM_SIZE


class HierarchicalDataGenerator(BaseDataGenerator):
    """Data generator for hierarchical OCR model"""
    
    def __init__(
        self,
        max_length=50,
        image_height=128,
        image_width=512,
        language="mixed",
        corpus_files=None,
        word_to_idx=None,
        ngram_to_idx=None,
    ):
        """
        Initialize the hierarchical data generator

        Args:
            max_length: Maximum text length
            image_height: Height of generated images
            image_width: Width of generated images
            language: Language to generate ('english', 'romanian', 'hungarian', or 'mixed')
            corpus_files: Dictionary mapping language names to corpus file paths
            word_to_idx: Word to index mapping
            ngram_to_idx: N-gram to index mapping
        """
        super().__init__(max_length, image_height, image_width, language, corpus_files)
        
        self.word_to_idx = word_to_idx or {"<PAD>": 0, "<UNK>": 1}
        self.ngram_to_idx = ngram_to_idx or {"<PAD>": 0, "<UNK>": 1}

    def text_to_multilevel_targets(self, text, language):
        """
        Convert text to character, n-gram, word, and language targets

        Args:
            text: Input text
            language: Text language

        Returns:
            Dictionary with targets for each level
        """
        # Character-level target
        char_target = [CHAR_TO_IDX.get(char, 0) for char in text]

        # Word-level target
        words = text.split()
        word_target = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>

        # N-gram level target
        ngram_target = []
        for i in range(len(text) - NGRAM_SIZE + 1):
            ngram = text[i : i + NGRAM_SIZE]
            ngram_target.append(self.ngram_to_idx.get(ngram, 1))  # 1 is <UNK>

        # Language target
        language_target = LANGUAGE_TO_IDX.get(language, 0)  # Default to English

        return {
            "char": char_target,
            "ngram": ngram_target,
            "word": word_target,
            "language": language_target,
        }

    def generate_sample(self, method="corpus"):
        """
        Generate a text sample with corresponding image and multilevel targets

        Args:
            method: Method to generate text

        Returns:
            Tuple of (text, image, language, multilevel_targets)
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

        # Generate multilevel targets
        multilevel_targets = self.text_to_multilevel_targets(text, selected_lang)

        return text, image, selected_lang, multilevel_targets


class HierarchicalOCRDataset(Dataset):
    def __init__(
        self, samples, transform=None, max_length=50, max_words=10, max_ngrams=48
    ):
        """
        OCR dataset from samples with hierarchical labeling

        Args:
            samples: List of (text, image, language, multilevel_targets) tuples
            transform: PyTorch image transforms
            max_length: Maximum text length for character-level
            max_words: Maximum number of words
            max_ngrams: Maximum number of n-grams
        """
        self.samples = samples
        self.transform = transform
        self.max_length = max_length
        self.max_words = max_words
        self.max_ngrams = max_ngrams

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, image, language, multilevel_targets = self.samples[idx]

        # Apply transforms if needed
        if self.transform:
            image = self.transform(image)

        # Character-level target
        char_target = multilevel_targets["char"]
        char_target = char_target + [0] * (self.max_length - len(char_target))  # Pad
        char_target = torch.LongTensor(
            char_target[: self.max_length]
        )  # Truncate if needed
        char_length = min(len(multilevel_targets["char"]), self.max_length)
        char_length = torch.LongTensor([char_length])

        # Word-level target
        word_target = multilevel_targets["word"]
        word_target = word_target + [0] * (self.max_words - len(word_target))  # Pad
        word_target = torch.LongTensor(
            word_target[: self.max_words]
        )  # Truncate if needed
        word_length = min(len(multilevel_targets["word"]), self.max_words)
        word_length = torch.LongTensor([word_length])

        # N-gram level target
        ngram_target = multilevel_targets["ngram"]
        ngram_target = ngram_target + [0] * (self.max_ngrams - len(ngram_target))  # Pad
        ngram_target = torch.LongTensor(
            ngram_target[: self.max_ngrams]
        )  # Truncate if needed
        ngram_length = min(len(multilevel_targets["ngram"]), self.max_ngrams)
        ngram_length = torch.LongTensor([ngram_length])

        # Language target
        language_target = torch.LongTensor([multilevel_targets["language"]])

        # Language one-hot encoding for feedback mechanism
        language_onehot = torch.zeros(3)  # 3 languages
        language_onehot[multilevel_targets["language"]] = 1

        return (
            image,
            {
                "char": char_target,
                "char_length": char_length,
                "word": word_target,
                "word_length": word_length,
                "ngram": ngram_target,
                "ngram_length": ngram_length,
                "language": language_target,
                "language_onehot": language_onehot,
            },
            language,
        )