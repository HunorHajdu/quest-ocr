"""
Corpus downloading and sampling utilities
"""

import os
import requests
import gzip
import shutil
import random
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def download_and_extract_file(url, output_dir="data"):
    """
    Download and extract a gzipped text file if it doesn't already exist.

    Args:
        url: URL of the gzipped text file
        output_dir: Directory to save the files

    Returns:
        Path to the extracted text file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename from the URL
    filename = url.split("/")[-1]
    base_filename = filename.replace(".gz", "")

    # Paths for the downloaded and extracted files
    download_path = os.path.join(output_dir, filename)
    extracted_path = os.path.join(output_dir, base_filename)

    # Check if the extracted file already exists
    if os.path.exists(extracted_path):
        logger.info(f"File {extracted_path} already exists, skipping download.")
        return extracted_path

    # Download the file if it doesn't exist
    if not os.path.exists(download_path):
        logger.info(f"Downloading {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded {download_path}")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    else:
        logger.info(f"File {download_path} already exists, skipping download.")

    # Extract the gzipped file
    try:
        logger.info(f"Extracting {download_path}...")
        with gzip.open(download_path, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Extracted to {extracted_path}")
        return extracted_path
    except Exception as e:
        logger.error(f"Failed to extract {download_path}: {e}")
        return None


def create_corpus_sampler(file_path, max_length=50, cache_size=1000):
    """
    Create a function that efficiently samples text from a large corpus file.
    Handles both short dialogue lines and longer text.

    Args:
        file_path: Path to the corpus text file
        max_length: Maximum length of text to sample
        cache_size: Number of lines to cache for efficient sampling

    Returns:
        Function that returns a random text sample
    """
    try:
        file_size = os.path.getsize(file_path)
        sample_cache = []

        def sample_text():
            nonlocal sample_cache

            # Refill cache if empty
            if not sample_cache:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        # Seek to a random position in the file
                        position = random.randint(0, max(0, file_size - 50000))
                        f.seek(position)

                        # Skip the current line to avoid starting in the middle of a line
                        f.readline()

                        # Store lines for processing
                        lines = []
                        for _ in range(
                            cache_size * 3
                        ):  # Read more lines than needed to construct good samples
                            line = f.readline().strip()
                            if line:  # Accept any non-empty line
                                lines.append(line)
                            if len(lines) >= cache_size * 3:
                                break

                        # Process the lines to create samples
                        i = 0
                        while i < len(lines):
                            # Strategy 1: Use single lines if they're long enough
                            if (
                                len(lines[i]) >= max_length * 0.5
                            ):  # Accept if at least half the max length
                                sample_cache.append(lines[i])
                                i += 1
                                continue

                            # Strategy 2: Combine consecutive short lines to create a sample of appropriate length
                            combined_text = lines[i]
                            j = i + 1
                            while (
                                j < len(lines) and len(combined_text) < max_length * 0.8
                            ):  # Try to get close to max length
                                combined_text += " " + lines[j]
                                j += 1
                                # If we've added at least 2 lines and have a decent length, consider it good enough
                                if j > i + 1 and len(combined_text) >= max_length * 0.3:
                                    break

                            sample_cache.append(combined_text)
                            i = j

                        # If we still don't have enough samples, add individual lines even if they're short
                        if len(sample_cache) < cache_size // 2:
                            for line in lines:
                                if line not in sample_cache:
                                    sample_cache.append(line)
                                if len(sample_cache) >= cache_size:
                                    break

                except Exception as e:
                    logger.error(f"Error reading from corpus file {file_path}: {e}")

            # If cache is still empty, return a fallback string
            if not sample_cache:
                logger.warning(f"Failed to sample text from {file_path}")
                return "Sample text unavailable"

            # Get a random sample from the cache
            text = sample_cache.pop(random.randint(0, len(sample_cache) - 1))

            # Truncate if necessary
            if len(text) > max_length:
                # Try to truncate at a word boundary
                truncate_pos = text.rfind(" ", 0, max_length)
                if truncate_pos > 0:
                    text = text[:truncate_pos]
                else:
                    text = text[:max_length]

            return text

        return sample_text

    except Exception as e:
        logger.error(f"Failed to create corpus sampler for {file_path}: {e}")
        # Return a function that returns a default string
        return lambda: "Sample text unavailable"


def create_corpus_samplers(corpus_files, max_length=50):
    """
    Create text samplers for each corpus file.

    Args:
        corpus_files: Dictionary mapping language names to file paths
        max_length: Maximum length of sampled text

    Returns:
        Dictionary mapping language names to sampler functions
    """
    samplers = {}

    for lang, file_path in corpus_files.items():
        if file_path and os.path.exists(file_path):
            samplers[lang] = create_corpus_sampler(file_path, max_length)
            logger.info(f"Created sampler for {lang} corpus")
        else:
            logger.warning(f"No valid corpus file for {lang}")

    return samplers


def download_corpus_files(urls, output_dir="data"):
    """
    Download and extract all corpus files.

    Args:
        urls: Dictionary mapping language names to URLs
        output_dir: Directory to save the files

    Returns:
        Dictionary mapping language names to file paths
    """
    corpus_files = {}

    for language, url in urls.items():
        logger.info(f"Processing {language} corpus...")
        file_path = download_and_extract_file(url, output_dir)
        if file_path:
            corpus_files[language] = file_path
        else:
            logger.warning(f"Failed to download/extract {language} corpus")

    return corpus_files


def build_vocabulary_from_corpus(corpus_files, max_vocab_size=10000, ngram_size=3):
    """
    Build word and n-gram vocabularies from corpus files
    (Used for hierarchical model)

    Args:
        corpus_files: Dictionary mapping language names to file paths
        max_vocab_size: Maximum vocabulary size
        ngram_size: Size of n-grams

    Returns:
        word_to_idx, idx_to_word, ngram_to_idx, idx_to_ngram
    """
    word_counts = Counter()
    ngram_counts = Counter()

    # Process each corpus
    for language, file_path in corpus_files.items():
        if not file_path or not os.path.exists(file_path):
            continue

        try:
            # Sample chunks randomly from corpus
            file_size = os.path.getsize(file_path)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Process 100 chunks of 10,000 chars each
                for _ in range(100):
                    position = random.randint(0, max(0, file_size - 10000))
                    f.seek(position)
                    f.readline()  # Skip partial line

                    text = f.read(10000)

                    # Process words
                    words = text.split()
                    word_counts.update(words)

                    # Process n-grams
                    for i in range(len(text) - ngram_size + 1):
                        ngram = text[i : i + ngram_size]
                        ngram_counts[ngram] += 1
        except Exception as e:
            logger.error(f"Error processing corpus {file_path}: {e}")

    # Build word vocabulary
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx_to_word = {0: "<PAD>", 1: "<UNK>"}

    for idx, (word, _) in enumerate(word_counts.most_common(max_vocab_size - 2)):
        word_idx = idx + 2  # Start after <PAD> and <UNK>
        word_to_idx[word] = word_idx
        idx_to_word[word_idx] = word

    # Build n-gram vocabulary
    ngram_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx_to_ngram = {0: "<PAD>", 1: "<UNK>"}

    max_ngram_vocab_size = 5000  # You might want to make this configurable
    for idx, (ngram, _) in enumerate(
        ngram_counts.most_common(max_ngram_vocab_size - 2)
    ):
        ngram_idx = idx + 2  # Start after <PAD> and <UNK>
        ngram_to_idx[ngram] = ngram_idx
        idx_to_ngram[ngram_idx] = ngram

    logger.info(f"Built word vocabulary of size {len(word_to_idx)}")
    logger.info(f"Built n-gram vocabulary of size {len(ngram_to_idx)}")

    return word_to_idx, idx_to_word, ngram_to_idx, idx_to_ngram