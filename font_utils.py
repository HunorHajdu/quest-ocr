"""
Font downloading and handling utilities
"""

import os
import requests
import shutil
import logging
from PIL import Image, ImageDraw, ImageFont
from config import FONT_DIR, FONT_URLS

logger = logging.getLogger(__name__)


def download_fonts():
    """Download multilingual fonts that support Eastern European characters"""
    for font_name, url in FONT_URLS.items():
        font_path = os.path.join(FONT_DIR, font_name)
        if not os.path.exists(font_path):
            try:
                logger.info(f"Downloading font: {font_name} from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(font_path, "wb") as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)

                logger.info(f"Font downloaded successfully: {font_path}")
            except Exception as e:
                logger.error(f"Failed to download font {font_name}: {e}")

    # Check if we successfully downloaded at least one font
    available_fonts = [
        f for f in os.listdir(FONT_DIR) if f.endswith((".ttf", ".TTF", ".otf", ".OTF"))
    ]
    if not available_fonts:
        logger.warning("No fonts downloaded. Using system default font.")
        return None

    # Return the path to the first available font
    return os.path.join(FONT_DIR, available_fonts[0])


def load_font(font_size=18):
    """
    Load a font that supports Eastern European characters
    
    Args:
        font_size: Size of the font to load
        
    Returns:
        PIL ImageFont object
    """
    # Download fonts if needed
    download_fonts()

    # List all available fonts in the font directory
    font_files = [
        f
        for f in os.listdir(FONT_DIR)
        if f.endswith((".ttf", ".TTF", ".otf", ".OTF"))
    ]
    font_paths = [os.path.join(FONT_DIR, f) for f in font_files]

    # If we have no fonts in the directory, use system default
    if not font_paths:
        logger.warning("No suitable fonts found. Using system default font.")
        return ImageFont.load_default()

    # Try to load each font until we find one that works
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            logger.info(f"Using font: {font_path}")

            # Test by rendering a few special characters
            test_text = "ăâîșțáéíóöőúüű"
            test_img = Image.new("L", (200, 30), color=255)
            draw = ImageDraw.Draw(test_img)
            draw.text((10, 10), test_text, fill=0, font=font)

            # If we get here without error, the font is working
            logger.info(
                f"Successfully loaded font {font_path} with special character support"
            )
            return font
        except Exception as e:
            logger.warning(f"Failed to use font {font_path}: {e}")

    # If all fonts fail, fall back to default
    logger.warning("All fonts failed to load. Using system default font.")
    return ImageFont.load_default()