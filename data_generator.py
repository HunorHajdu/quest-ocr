"""
Base data generation utilities shared between simple and hierarchical models
"""

import random
import logging
import numpy as np
from PIL import Image, ImageDraw
from corpus_utils import create_corpus_samplers
from font_utils import load_font
from config import CHAR_TO_IDX, LANGUAGE_TO_IDX

logger = logging.getLogger(__name__)


class BaseDataGenerator:
    """Base class for data generation with common functionality"""
    
    def __init__(
        self,
        max_length=50,
        image_height=128,
        image_width=512,
        language="mixed",
        corpus_files=None,
    ):
        """
        Initialize the base data generator

        Args:
            max_length: Maximum text length
            image_height: Height of generated images
            image_width: Width of generated images
            language: Language to generate ('english', 'romanian', 'hungarian', or 'mixed')
            corpus_files: Dictionary mapping language names to corpus file paths
        """
        self.language = language
        self.max_length = max_length
        self.image_height = image_height
        self.image_width = image_width
        self.corpus_files = corpus_files or {}

        # Default to mixed if not specified correctly
        if language not in ["english", "romanian", "hungarian", "mixed"]:
            self.language = "mixed"

        logger.info(f"Initialized data generator for {self.language} language(s)")

        # Initialize corpus samplers if corpus files are provided
        self.corpus_samplers = {}
        if self.corpus_files:
            self.corpus_samplers = create_corpus_samplers(
                self.corpus_files, self.max_length
            )
            logger.info(
                f"Created corpus samplers for: {list(self.corpus_samplers.keys())}"
            )

        # Load font
        self.font = load_font(font_size=18)

    def generate_text(self, method="corpus"):
        """
        Generate random text using corpus data

        Args:
            method: The method to generate text ("corpus" or "mixed")

        Returns:
            A string of generated text
        """
        # For mixed language, select a specific language for this text
        if self.language == "mixed":
            selected_lang = random.choice(["english", "romanian", "hungarian"])
        else:
            selected_lang = self.language

        # Use corpus if available
        if selected_lang in self.corpus_samplers:
            try:
                # Attempt to get a suitable text sample from corpus
                # Try up to 3 times to get a sample of appropriate length
                for _ in range(3):
                    text = self.corpus_samplers[selected_lang]()

                    # Check if we got a valid sample
                    if text and text != "Sample text unavailable":
                        # For OCR training, prefer samples that are neither too short nor too long
                        # But still accept various lengths to ensure diversity
                        if (
                            len(text) > 10 or random.random() < 0.3
                        ):  # Always accept longer texts, sometimes accept shorter ones
                            return text

                # If we couldn't get a good sample after several tries, use the last one we got
                if text and text != "Sample text unavailable":
                    return text

                # Fallback if all attempts failed
                logger.warning(
                    f"Corpus sampler for {selected_lang} failed to return valid text, using default"
                )
                return f"Sample {selected_lang} text"

            except Exception as e:
                logger.warning(f"Error generating text from corpus: {e}, using default")
                return f"Sample {selected_lang} text"
        else:
            logger.warning(f"No corpus sampler for {selected_lang}, using default")
            return f"Sample {selected_lang} text"

    def render_text_to_image(self, text):
        """
        Render text to an image with enhanced augmentation

        Args:
            text: The text to render

        Returns:
            PIL Image with rendered text
        """
        # Create blank image
        img = Image.new("L", (self.image_width, self.image_height), color=255)
        draw = ImageDraw.Draw(img)

        # Calculate text size and position using modern Pillow methods
        try:
            # For Pillow >= 9.2.0
            left, top, right, bottom = draw.textbbox((0, 0), text, font=self.font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            try:
                # For Pillow >= 8.0.0
                left, top, right, bottom = self.font.getbbox(text)
                text_width = right - left
                text_height = bottom - top
            except AttributeError:
                # Fallback for older Pillow versions - approximate dimensions
                text_width = (
                    self.font.getlength(text)
                    if hasattr(self.font, "getlength")
                    else len(text) * 10
                )
                text_height = self.image_height // 2

        # Center text
        position = (
            (self.image_width - text_width) // 2,
            (self.image_height - text_height) // 2,
        )

        # Draw text
        draw.text(position, text, fill=0, font=self.font)

        # Apply random transformations for data augmentation
        img = self._apply_augmentations(img)

        return img

    def _apply_augmentations(self, img):
        """Apply data augmentation transformations to the image"""
        # 1. Random rotation
        if random.random() > 0.3:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 2. Random noise
        if random.random() > 0.3:
            noise_level = random.uniform(5, 15)
            noise = np.random.normal(
                0, noise_level, (self.image_height, self.image_width)
            )
            img_array = np.array(img)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        # 3. Random perspective distortion
        if random.random() > 0.7:
            img = self._apply_perspective_distortion(img)

        # 4. Random blur
        if random.random() > 0.8:
            img = self._apply_blur(img)

        return img

    def _apply_perspective_distortion(self, img):
        """Apply perspective distortion to the image"""
        try:
            width, height = img.size
            # Generate random distortion points
            distortion = random.uniform(0.02, 0.05)
            topleft = (
                random.uniform(0, distortion * width),
                random.uniform(0, distortion * height),
            )
            topright = (
                random.uniform(width * (1 - distortion), width),
                random.uniform(0, distortion * height),
            )
            bottomright = (
                random.uniform(width * (1 - distortion), width),
                random.uniform(height * (1 - distortion), height),
            )
            bottomleft = (
                random.uniform(0, distortion * width),
                random.uniform(height * (1 - distortion), height),
            )

            # Apply perspective transform
            coeffs = self._find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [topleft, topright, bottomright, bottomleft],
            )
            img = img.transform(
                (width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC
            )
        except (AttributeError, ImportError, NameError):
            # Skip if transform is not available
            pass
        return img

    def _apply_blur(self, img):
        """Apply blur to the image"""
        try:
            from PIL import ImageFilter
            blur_radius = random.uniform(0.2, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except (ImportError, AttributeError):
            # Skip if ImageFilter is not available
            pass
        return img

    def _find_coeffs(self, source_coords, target_coords):
        """Find coefficients for perspective transform"""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])

        A = np.matrix(matrix, dtype=float)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def generate_sample(self, method="corpus"):
        """
        Generate a text sample with corresponding image
        This method should be overridden in subclasses

        Args:
            method: Method to generate text

        Returns:
            Tuple containing generated data
        """
        raise NotImplementedError("Subclasses must implement generate_sample")

    def generate_batch(self, batch_size=32, method="corpus"):
        """
        Generate a batch of samples

        Args:
            batch_size: Number of samples to generate
            method: Text generation method

        Returns:
            List of sample tuples
        """
        samples = []
        for _ in range(batch_size):
            sample = self.generate_sample(method=method)
            samples.append(sample)
        return samples