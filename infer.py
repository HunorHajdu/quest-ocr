"""
Inference script for OCR models
Provides a simple interface for loading trained models and performing OCR on images
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging

# Import models
from quest_ocr.models import OCRModel, HierarchicalOCRModel
from quest_ocr.config import IDX_TO_CHAR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRInference:
    """
    Unified inference class for both simple and hierarchical OCR models
    """

    def __init__(self, model_path, device=None, model_type=None):
        """
        Initialize the OCR inference system

        Args:
            model_path: Path to the trained model file (.pth)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
            model_type: Force model type ('simple', 'hierarchical', or None for auto-detect)
        """
        self.model_path = model_path
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.force_model_type = model_type

        # Load model and metadata
        self.model, self.model_type, self.architecture, self.metadata = (
            self._load_model()
        )

        # Setup image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 512)),  # Standard OCR image size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        logger.info(
            f"Loaded {self.model_type} OCR model ({self.architecture}) on {self.device}"
        )

    def _detect_model_type(self, checkpoint):
        """
        Auto-detect model type from checkpoint

        Args:
            checkpoint: Loaded model checkpoint

        Returns:
            str: Detected model type ('simple' or 'hierarchical')
        """
        # First check if model_type is explicitly stored
        if "model_type" in checkpoint:
            return checkpoint["model_type"]

        # Auto-detect based on state_dict keys
        state_dict_keys = checkpoint["model_state_dict"].keys()

        # Look for hierarchical model signatures
        hierarchical_keys = [
            "char_decoder",
            "ngram_decoder",
            "word_decoder",
            "char_embedding",
            "ngram_embedding",
            "word_embedding",
            "char_attention",
            "ngram_attention",
        ]

        # Look for simple model signatures
        simple_keys = ["fc.weight", "fc.bias"]

        # Check for hierarchical model keys
        if any(
            any(hkey in key for key in state_dict_keys) for hkey in hierarchical_keys
        ):
            logger.info("Auto-detected hierarchical model from state_dict keys")
            return "hierarchical"

        # Check for simple model keys
        if any(any(skey in key for key in state_dict_keys) for skey in simple_keys):
            logger.info("Auto-detected simple model from state_dict keys")
            return "simple"

        # Fallback: check vocabulary sizes
        if "word_to_idx" in checkpoint and len(checkpoint["word_to_idx"]) > 2:
            logger.info("Auto-detected hierarchical model from vocabulary presence")
            return "hierarchical"

        # Default fallback
        logger.warning("Could not auto-detect model type, defaulting to simple")
        return "simple"

    def _load_model(self):
        """Load the trained model from file with auto-detection"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Determine model type
        if self.force_model_type:
            model_type = self.force_model_type
            logger.info(f"Using forced model type: {model_type}")
        else:
            model_type = self._detect_model_type(checkpoint)

        # Extract architecture
        architecture = checkpoint.get("architecture", "lstm")

        # Create model based on detected/forced type
        if model_type == "simple":
            model = OCRModel(
                input_channels=1,
                architecture=architecture,
                num_layers=4,
                hidden_size=512,
            )
        elif model_type == "hierarchical":
            # Get vocabulary sizes with fallbacks
            char_vocab_size = len(checkpoint.get("char_to_idx", IDX_TO_CHAR))
            word_vocab_size = len(
                checkpoint.get("word_to_idx", {"<PAD>": 0, "<UNK>": 1})
            )
            ngram_vocab_size = len(
                checkpoint.get("ngram_to_idx", {"<PAD>": 0, "<UNK>": 1})
            )

            model = HierarchicalOCRModel(
                input_channels=1,
                architecture=architecture,
                num_chars=char_vocab_size,
                num_words=word_vocab_size,
                num_ngrams=ngram_vocab_size,
                num_layers=4,
                hidden_size=512,
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Must be 'simple' or 'hierarchical'"
            )

        # Load model weights with error handling
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Successfully loaded model weights")
        except RuntimeError as e:
            logger.error(f"Error loading model weights: {e}")
            logger.info("This might indicate a model type mismatch.")
            logger.info(
                "Try explicitly setting model_type='simple' or model_type='hierarchical'"
            )
            raise

        model.to(self.device)
        model.eval()

        # Extract additional metadata
        metadata = {
            "char_to_idx": checkpoint.get("char_to_idx", {}),
            "idx_to_char": checkpoint.get("idx_to_char", IDX_TO_CHAR),
            "word_to_idx": checkpoint.get("word_to_idx", {}),
            "ngram_to_idx": checkpoint.get("ngram_to_idx", {}),
            "performance": checkpoint.get("performance", {}),
        }

        return model, model_type, architecture, metadata

    def _preprocess_image(self, image):
        """
        Preprocess image for OCR

        Args:
            image: PIL Image or path to image file

        Returns:
            Preprocessed tensor ready for model input
        """
        # Handle different input types
        if isinstance(image, str):
            # Load image from file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or file path")

        # Convert to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Apply transforms and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        return image_tensor

    def _decode_predictions(self, outputs):
        """
        Decode model predictions to text using CTC decoding

        Args:
            outputs: Model outputs (log probabilities)

        Returns:
            Decoded text string
        """
        # Get the most likely character at each step
        _, indices = torch.max(outputs, dim=2)
        indices = indices.cpu().numpy()[0]  # Remove batch dimension

        # CTC decoding: remove duplicates and blank tokens
        decoded_chars = []
        prev_idx = -1

        for idx in indices:
            if idx != prev_idx and idx != 0:  # Skip if same as previous or blank token
                decoded_chars.append(self.metadata["idx_to_char"].get(idx, ""))
            prev_idx = idx

        return "".join(decoded_chars)

    def predict(self, image, return_confidence=False):
        """
        Perform OCR on an image

        Args:
            image: PIL Image or path to image file
            return_confidence: If True, also return confidence scores

        Returns:
            predicted_text (str): The recognized text
            confidence (float, optional): Average confidence score if return_confidence=True
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)

        # Run inference
        with torch.no_grad():
            if self.model_type == "simple":
                outputs = self.model(image_tensor)
                predicted_text = self._decode_predictions(outputs)

                # Calculate confidence if requested
                if return_confidence:
                    probs = torch.exp(outputs)  # Convert log probs to probs
                    max_probs, _ = torch.max(probs, dim=2)
                    confidence = torch.mean(max_probs).item()
                    return predicted_text, confidence

            else:  # hierarchical
                outputs = self.model(image_tensor)
                # Use character-level predictions for text output
                predicted_text = self._decode_predictions(outputs["char"])

                # Calculate confidence if requested
                if return_confidence:
                    probs = torch.exp(outputs["char"])  # Convert log probs to probs
                    max_probs, _ = torch.max(probs, dim=2)
                    confidence = torch.mean(max_probs).item()
                    return predicted_text, confidence

        return predicted_text

    def predict_batch(self, images, return_confidence=False):
        """
        Perform OCR on a batch of images

        Args:
            images: List of PIL Images or image file paths
            return_confidence: If True, also return confidence scores

        Returns:
            List of predicted texts (and confidences if requested)
        """
        results = []

        for image in images:
            if return_confidence:
                text, confidence = self.predict(image, return_confidence=True)
                results.append((text, confidence))
            else:
                text = self.predict(image, return_confidence=False)
                results.append(text)

        return results

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "device": str(self.device),
            "model_path": self.model_path,
            "performance": self.metadata.get("performance", {}),
        }

        if self.model_type == "hierarchical":
            info["vocab_sizes"] = {
                "characters": len(self.metadata["char_to_idx"]),
                "words": len(self.metadata["word_to_idx"]),
                "ngrams": len(self.metadata["ngram_to_idx"]),
            }
        else:
            info["vocab_sizes"] = {
                "characters": len(self.metadata["char_to_idx"]),
            }

        return info


def load_ocr_model(model_path, device=None, model_type=None):
    """
    Convenience function to load an OCR model for inference

    Args:
        model_path: Path to the trained model file (.pth)
        device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        model_type: Force model type ('simple', 'hierarchical', or None for auto-detect)

    Returns:
        OCRInference instance ready for prediction
    """
    return OCRInference(model_path, device, model_type)


def predict_text(
    model_path, image, device=None, return_confidence=False, model_type=None
):
    """
    Convenience function for one-shot OCR prediction

    Args:
        model_path: Path to the trained model file (.pth)
        image: PIL Image or path to image file
        device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        return_confidence: If True, also return confidence score
        model_type: Force model type ('simple', 'hierarchical', or None for auto-detect)

    Returns:
        predicted_text (str): The recognized text
        confidence (float, optional): Confidence score if return_confidence=True
    """
    ocr = OCRInference(model_path, device, model_type)
    return ocr.predict(image, return_confidence)


# Example usage and testing functions
def demo_inference(model_path, image_path):
    """
    Demonstration of how to use the inference system

    Args:
        model_path: Path to trained model
        image_path: Path to test image
    """
    print("=" * 60)
    print("OCR INFERENCE DEMO")
    print("=" * 60)

    try:
        # Load the OCR model
        print(f"Loading model from: {model_path}")
        ocr = load_ocr_model(model_path)

        # Display model information
        model_info = ocr.get_model_info()
        print(f"\nModel Information:")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Vocabulary sizes: {model_info['vocab_sizes']}")

        if model_info["performance"]:
            print(
                f"  Performance metrics available: {list(model_info['performance'].keys())}"
            )

        # Perform OCR prediction
        print(f"\nPerforming OCR on: {image_path}")

        # Predict with confidence
        predicted_text, confidence = ocr.predict(image_path, return_confidence=True)

        print(f"\nResults:")
        print(f"  Predicted text: '{predicted_text}'")
        print(f"  Confidence: {confidence:.4f}")

        # Test batch prediction
        print(f"\nTesting batch prediction with same image...")
        batch_results = ocr.predict_batch(
            [image_path, image_path], return_confidence=True
        )

        for i, (text, conf) in enumerate(batch_results):
            print(f"  Batch {i + 1}: '{text}' (confidence: {conf:.4f})")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    """
    Example usage of the inference script
    """
    import sys

    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        demo_inference(model_path, image_path)
    else:
        print("Usage: python infer.py <model_path> <image_path>")
        print("\nExample:")
        print(
            "python infer.py results_simple_20241201_120000/simple_lstm/models/simple_ocr_model_lstm.pth test_image.png"
        )
        print("\nAlternatively, you can use the inference functions in your own code:")
        print("""
            # Simple usage with auto-detection
            from infer import predict_text
            text = predict_text("path/to/model.pth", "path/to/image.png")

            # Force model type
            text = predict_text("path/to/model.pth", "path/to/image.png", model_type="hierarchical")

            # Advanced usage
            from infer import load_ocr_model
            ocr = load_ocr_model("path/to/model.pth", model_type="simple")
            text, confidence = ocr.predict("path/to/image.png", return_confidence=True)
            batch_results = ocr.predict_batch(["img1.png", "img2.png"])
        """)
