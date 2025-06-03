"""
Configuration file for OCR system
Contains all constants and shared configurations
"""

import string
import os

# Character sets and mappings
ENGLISH_CHARS = string.ascii_letters + string.digits + string.punctuation + " "
ROMANIAN_CHARS = "ăâîșțĂÂÎȘȚ"
HUNGARIAN_CHARS = "áéíóöőúüűÁÉÍÓÖŐÚÜŰ"
CHAR_SET = ENGLISH_CHARS + ROMANIAN_CHARS + HUNGARIAN_CHARS

CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHAR_SET)}
CHAR_TO_IDX["<PAD>"] = 0  # Padding token
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CHARS = len(CHAR_TO_IDX)

# For hierarchical model - word-level prediction
WORD_TO_IDX = {"<PAD>": 0, "<UNK>": 1}  # Will be dynamically populated
IDX_TO_WORD = {0: "<PAD>", 1: "<UNK>"}
MAX_VOCAB_SIZE = 10000

# For hierarchical model - n-gram level prediction
NGRAM_SIZE = 3
NGRAM_TO_IDX = {"<PAD>": 0, "<UNK>": 1}  # Will be dynamically populated
IDX_TO_NGRAM = {0: "<PAD>", 1: "<UNK>"}
MAX_NGRAM_VOCAB_SIZE = 5000

# Language mapping
LANGUAGE_CODES = {"english": "en", "romanian": "ro", "hungarian": "hu"}
LANGUAGE_TO_IDX = {"english": 0, "romanian": 1, "hungarian": 2}
IDX_TO_LANGUAGE = {0: "english", 1: "romanian", 2: "hungarian"}

# Font settings
FONT_DIR = "fonts"
if not os.path.exists(FONT_DIR):
    os.makedirs(FONT_DIR)

# List of fonts that support Eastern European characters
FONT_URLS = {
    "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "DejaVuSans.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "Arimo-Regular.ttf": "https://github.com/googlefonts/Arimo/raw/main/fonts/ttf/Arimo-Regular.ttf",
}

# Corpus URLs
CORPUS_URLS = {
    "hungarian": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/hu.txt.gz",
    "romanian": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/ro.txt.gz",
    "english": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/en.txt.gz",
}

# Training parameters
DEFAULT_PARAMS = {
    "max_length": 50,
    "image_height": 128,
    "image_width": 512,
    "epochs": 500,
    "language": "mixed",  # 'english', 'romanian', 'hungarian', or 'mixed'
}

# Simple model parameters
SIMPLE_PARAMS = {
    **DEFAULT_PARAMS,
    "batch_size": 320,
    "train_samples": 1200,
    "val_samples": 120,
}

# Hierarchical model parameters
HIERARCHICAL_PARAMS = {
    **DEFAULT_PARAMS,
    "batch_size": 64,  # Reduced for more memory-intensive hierarchical model
    "train_samples": 10000,
    "val_samples": 1000,
    "epochs": 50,  # Fewer epochs for hierarchical model
}

# Available architectures
ARCHITECTURES = ["rnn", "gru", "lstm", "transformer"]