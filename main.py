"""
Main execution file for OCR system with model switching
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime, timedelta

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import configuration
from config import (
    CORPUS_URLS,
    SIMPLE_PARAMS,
    HIERARCHICAL_PARAMS,
    ARCHITECTURES,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    MAX_VOCAB_SIZE,
)

# Import utilities
from corpus_utils import download_corpus_files, build_vocabulary_from_corpus

# Import models
from models import OCRModel, HierarchicalOCRModel

# Import datasets
from dataset_generators import (
    CorpusDataGenerator,
    OCRDataset,
    HierarchicalDataGenerator,
    HierarchicalOCRDataset,
)

# Import trainers
from trainers import OCRTrainer, HierarchicalOCRTrainer

# Import utilities
from utils.visualization import test_and_visualize
from utils.evaluation import evaluate_model_metrics, evaluate_hierarchical_model_metrics

# Configure logger
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"ocr_training_{timestamp}.log")

# Configure root logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

logger.info(f"Logging to file: {log_file}")

# ============================================================================
# MODEL SELECTION FLAG - CHANGE THIS TO SWITCH BETWEEN MODELS
# ============================================================================
MODEL_TYPE = "simple"  # Options: "simple" or "hierarchical"
# ============================================================================


def setup_simple_model_components(corpus_files):
    """Setup components for simple OCR model"""
    params = SIMPLE_PARAMS

    # Create data generator
    data_gen = CorpusDataGenerator(
        max_length=params["max_length"],
        image_height=params["image_height"],
        image_width=params["image_width"],
        language=params["language"],
        corpus_files=corpus_files,
    )

    return data_gen, params


def setup_hierarchical_model_components(corpus_files):
    """Setup components for hierarchical OCR model"""
    params = HIERARCHICAL_PARAMS

    # Build vocabulary from corpus
    print("Building vocabulary from corpus files...")
    word_to_idx, idx_to_word, ngram_to_idx, idx_to_ngram = build_vocabulary_from_corpus(
        corpus_files, max_vocab_size=MAX_VOCAB_SIZE
    )

    print(f"Word vocabulary size: {len(word_to_idx)}")
    print(f"N-gram vocabulary size: {len(ngram_to_idx)}")

    # Create data generator
    data_gen = HierarchicalDataGenerator(
        max_length=params["max_length"],
        image_height=params["image_height"],
        image_width=params["image_width"],
        language=params["language"],
        corpus_files=corpus_files,
        word_to_idx=word_to_idx,
        ngram_to_idx=ngram_to_idx,
    )

    return data_gen, params, word_to_idx, idx_to_word, ngram_to_idx, idx_to_ngram


def create_model_and_trainer(architecture, model_type, additional_params=None):
    """Create model and trainer based on type and architecture"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_type == "simple":
        params = SIMPLE_PARAMS
        print(f"Creating simple model with {architecture} architecture...")
        model = OCRModel(
            input_channels=1, architecture=architecture, num_layers=4, hidden_size=512
        )
        trainer = OCRTrainer(model, max_length=params["max_length"], device=device)

    elif model_type == "hierarchical":
        params = HIERARCHICAL_PARAMS
        word_to_idx = additional_params["word_to_idx"]
        ngram_to_idx = additional_params["ngram_to_idx"]

        print(f"Creating hierarchical model with {architecture} architecture...")
        print(f"Creating hierarchical model with {architecture} architecture...")
        model = HierarchicalOCRModel(
            input_channels=1,
            architecture=architecture,
            num_chars=len(CHAR_TO_IDX),
            num_words=len(word_to_idx),
            num_ngrams=len(ngram_to_idx),
            num_layers=4,
            hidden_size=512,
        )
        trainer = HierarchicalOCRTrainer(
            model, max_length=params["max_length"], device=device
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, trainer, params


def create_datasets_and_loaders(data_gen, params, model_type):
    """Create datasets and data loaders"""
    print("Generating training and validation data...")
    data_gen_start = time.time()

    train_data = data_gen.generate_batch(
        batch_size=params["train_samples"], method="corpus"
    )
    val_data = data_gen.generate_batch(
        batch_size=params["val_samples"], method="corpus"
    )

    data_gen_time = time.time() - data_gen_start
    print(f"Data generation completed in {str(timedelta(seconds=int(data_gen_time)))}")

    # Print some statistics about the generated data
    language_counts = {}
    for sample in train_data:
        # Handle both simple and hierarchical sample formats
        if len(sample) == 3:  # Simple: (text, image, lang)
            _, _, lang = sample
        else:  # Hierarchical: (text, image, lang, multilevel_targets)
            _, _, lang, _ = sample
        language_counts[lang] = language_counts.get(lang, 0) + 1

    print("\nTraining data language distribution:")
    for lang, count in language_counts.items():
        print(f"  {lang}: {count} samples ({count / len(train_data) * 100:.1f}%)")

    # Create transform
    transform = transforms.Compose(
        [
            transforms.Resize((params["image_height"], params["image_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Create datasets
    if model_type == "simple":
        train_dataset = OCRDataset(
            train_data, transform=transform, max_length=params["max_length"]
        )
        val_dataset = OCRDataset(
            val_data, transform=transform, max_length=params["max_length"]
        )
    else:  # hierarchical
        train_dataset = HierarchicalOCRDataset(
            train_data, transform=transform, max_length=params["max_length"]
        )
        val_dataset = HierarchicalOCRDataset(
            val_data, transform=transform, max_length=params["max_length"]
        )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    return train_dataloader, val_dataloader


def save_model(
    model, trainer, architecture, model_type, model_path, additional_data=None
):
    """Save model with all necessary information"""
    save_data = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "model_type": model_type,
        "char_to_idx": CHAR_TO_IDX,
        "idx_to_char": IDX_TO_CHAR,
    }

    if additional_data:
        save_data.update(additional_data)

    torch.save(save_data, model_path)
    print(f"Model saved successfully to {model_path}")


def load_model(model_path, architecture, model_type, device):
    """Load model from file"""
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "simple":
        model = OCRModel(
            input_channels=1, architecture=architecture, num_layers=4, hidden_size=512
        )
        trainer = OCRTrainer(model, max_length=50, device=device)

    else:  # hierarchical
        model = HierarchicalOCRModel(
            input_channels=1,
            architecture=architecture,
            num_chars=len(checkpoint["char_to_idx"]),
            num_words=len(checkpoint["word_to_idx"]),
            num_ngrams=len(checkpoint["ngram_to_idx"]),
            num_layers=4,
            hidden_size=512,
        )
        trainer = HierarchicalOCRTrainer(model, max_length=50, device=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, trainer, checkpoint


def plot_training_history(history, architecture, model_type, plots_dir):
    """Plot and save training history"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 8))

        plt.subplot(2, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")

        if model_type == "simple":
            plt.subplot(2, 2, 2)
            plt.plot(history["overall_cer"], label="Overall CER")
            plt.xlabel("Epoch")
            plt.ylabel("CER")
            plt.legend()
            plt.title("Overall Character Error Rate")

            plt.subplot(2, 2, 3)
            plt.plot(history["english_cer"], label="English")
            plt.plot(history["romanian_cer"], label="Romanian")
            plt.plot(history["hungarian_cer"], label="Hungarian")
            plt.xlabel("Epoch")
            plt.ylabel("CER")
            plt.legend()
            plt.title("Character Error Rate by Language")

        else:  # hierarchical
            plt.subplot(2, 2, 2)
            plt.plot(history["overall_f1"], label="Overall F1")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.title("Overall F1 Score")

            plt.subplot(2, 2, 3)
            plt.plot(history["english_f1"], label="English")
            plt.plot(history["romanian_f1"], label="Romanian")
            plt.plot(history["hungarian_f1"], label="Hungarian")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.title("F1 Score by Language")

        # Add epoch timing plot
        if "epoch_times" in history and history["epoch_times"]:
            plt.subplot(2, 2, 4)
            plt.plot(history["epoch_times"], label="Epoch Time")
            plt.xlabel("Epoch")
            plt.ylabel("Time (seconds)")
            plt.title("Training Time per Epoch")

            # Add trend line with error handling
            try:
                if len(history["epoch_times"]) > 1:
                    import numpy as np

                    x = np.array(range(len(history["epoch_times"])))
                    y = np.array(history["epoch_times"])

                    # Filter out any NaN or Inf values
                    mask = np.isfinite(y)
                    if np.sum(mask) > 1:
                        z = np.polyfit(x[mask], y[mask], 1)
                        p = np.poly1d(z)
                        plt.plot(x[mask], p(x[mask]), "r--", label="Trend")

                plt.legend()
            except Exception as e:
                print(f"Could not calculate trend line: {e}")

        plt.tight_layout()
        history_plot_path = os.path.join(
            plots_dir, f"training_history_{architecture}.png"
        )
        plt.savefig(history_plot_path, dpi=150)
        plt.close()
        print(f"Training history plot saved to {history_plot_path}")

    except Exception as e:
        print(f"Error creating training history plot: {e}")


def main(should_train=True, specific_arch=None, path=None):
    """
    Main function to train or test the OCR model

    Args:
        should_train: Whether to train the model or just load and test
        specific_arch: If provided, only train/test this specific architecture
        path: Path to existing results directory for testing
    """
    # Start timing the entire process
    total_start_time = time.time()

    # Create a base results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = f"results_{MODEL_TYPE}_{timestamp}" if should_train else path
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
        print(f"Created results directory: {base_results_dir}")

    print(f"\n{'=' * 60}")
    print(f"RUNNING OCR SYSTEM - MODEL TYPE: {MODEL_TYPE.upper()}")
    print(f"{'=' * 60}")

    # Track performance metrics for each architecture
    performance_metrics = {}

    # List of architectures to train
    architectures = ARCHITECTURES

    # If specific_arch is provided, only train/test that one
    if specific_arch:
        if specific_arch in architectures:
            architectures = [specific_arch]
        else:
            print(
                f"Unknown architecture '{specific_arch}'. Must be one of: {', '.join(architectures)}"
            )
            return

    # Download corpus files
    print("Downloading and extracting corpus files...")
    corpus_files = download_corpus_files(CORPUS_URLS)

    # Setup model-specific components
    if MODEL_TYPE == "simple":
        data_gen, params = setup_simple_model_components(corpus_files)
        additional_model_params = None
    elif MODEL_TYPE == "hierarchical":
        data_gen, params, word_to_idx, idx_to_word, ngram_to_idx, idx_to_ngram = (
            setup_hierarchical_model_components(corpus_files)
        )
        additional_model_params = {
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "ngram_to_idx": ngram_to_idx,
            "idx_to_ngram": idx_to_ngram,
        }

        # Save vocabulary for later use
        vocab_dir = os.path.join(base_results_dir, "vocabulary")
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        with open(os.path.join(vocab_dir, "word_to_idx.json"), "w") as f:
            json.dump({str(k): v for k, v in word_to_idx.items()}, f)

        with open(os.path.join(vocab_dir, "ngram_to_idx.json"), "w") as f:
            json.dump({str(k): v for k, v in ngram_to_idx.items()}, f)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # Test the corpus samplers
    print(f"\nTesting corpus samplers for {MODEL_TYPE} model:")
    for lang in corpus_files.keys():
        print(f"\n{lang.capitalize()} samples:")
        data_gen.language = lang
        for i in range(3):
            sample_text = data_gen.generate_text()
            print(f"  Sample {i + 1}: {sample_text}")

    # Reset language to mixed
    data_gen.language = params["language"]

    # Generate data once and reuse for all architectures
    if should_train:
        train_dataloader, val_dataloader = create_datasets_and_loaders(
            data_gen, params, MODEL_TYPE
        )

    # Loop through each architecture
    for architecture in architectures:
        # Start timing this architecture
        arch_start_time = time.time()

        # Create architecture-specific directory
        arch_dir = os.path.join(base_results_dir, f"{MODEL_TYPE}_{architecture}")
        if not os.path.exists(arch_dir):
            os.makedirs(arch_dir)
            print(f"Created architecture directory: {arch_dir}")

        # Create subdirectories for different types of outputs
        models_dir = os.path.join(arch_dir, "models")
        plots_dir = os.path.join(arch_dir, "plots")
        samples_dir = os.path.join(arch_dir, "samples")
        metrics_dir = os.path.join(arch_dir, "metrics")

        for directory in [models_dir, plots_dir, samples_dir, metrics_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Define model path in the new directory structure
        model_path = os.path.join(
            models_dir, f"{MODEL_TYPE}_ocr_model_{architecture}.pth"
        )

        print(f"\n{'=' * 50}")
        print(f"{MODEL_TYPE.upper()} ARCHITECTURE: {architecture.upper()}")
        print(f"{'=' * 50}")

        # Create model and trainer
        model, trainer, current_params = create_model_and_trainer(
            architecture, MODEL_TYPE, additional_model_params
        )

        if should_train:
            # Train model
            print("Starting training...")
            train_start_time = time.time()

            history = trainer.train(
                train_dataloader, val_dataloader, epochs=current_params["epochs"]
            )

            train_time = time.time() - train_start_time
            print(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

            # Save model immediately after training completes
            print(f"\nSaving model to {model_path}...")

            save_data = {
                "performance": {
                    "train_time": train_time,
                    "avg_epoch_time": sum(history["epoch_times"])
                    / len(history["epoch_times"])
                    if history["epoch_times"]
                    else 0,
                }
            }

            if MODEL_TYPE == "simple":
                save_data["performance"]["final_cer"] = (
                    history["overall_cer"][-1] if history["overall_cer"] else 1.0
                )
            else:
                save_data["performance"]["final_f1"] = (
                    history["overall_f1"][-1] if history["overall_f1"] else 0.0
                )

            if additional_model_params:
                save_data.update(additional_model_params)

            save_model(model, trainer, architecture, MODEL_TYPE, model_path, save_data)

            # Plot training history
            plot_training_history(history, architecture, MODEL_TYPE, plots_dir)

            # Initialize performance metrics for this architecture
            performance_metrics[architecture] = save_data["performance"].copy()

            print(f"Training for {architecture} complete!")

        else:
            # Load the model instead of training
            print(f"Loading model from {model_path}...")
            try:
                model, trainer, checkpoint = load_model(
                    model_path, architecture, MODEL_TYPE, trainer.device
                )

                # Initialize performance metrics if loading model
                if "performance" in checkpoint:
                    performance_metrics[architecture] = checkpoint["performance"]
                else:
                    performance_metrics[architecture] = {}

                print("Model loaded successfully.")
            except FileNotFoundError:
                print(f"Model file {model_path} not found!")
                print(
                    "Please train the model first (should_train=True) or specify the correct path."
                )
                continue  # Skip to the next architecture
            except Exception as e:
                print(f"Error loading model: {e}")
                continue  # Skip to the next architecture

        # Always evaluate model metrics regardless of train/test mode
        print(f"Evaluating metrics for {architecture}...")
        try:
            # Choose evaluation function based on model type
            if MODEL_TYPE == "simple":
                metrics = evaluate_model_metrics(
                    trainer, data_gen, architecture, metrics_dir, num_samples=100
                )
            else:  # hierarchical
                metrics = evaluate_hierarchical_model_metrics(
                    trainer, data_gen, architecture, metrics_dir, num_samples=100
                )

            # Add metrics to performance_metrics
            if architecture in performance_metrics:
                performance_metrics[architecture].update(metrics)
            else:
                performance_metrics[architecture] = metrics

        except Exception as e:
            print(f"Error evaluating metrics: {e}")
            traceback.print_exc()

        # Test model and visualize results
        try:
            # Start timing testing
            test_start_time = time.time()

            # Generate more comprehensive testing with visualization
            test_and_visualize(
                trainer,
                data_gen,
                architecture,
                num_samples=50 if MODEL_TYPE == "hierarchical" else 100,
                output_dir=arch_dir,
            )

            # Record testing time
            test_time = time.time() - test_start_time
            if architecture in performance_metrics:
                performance_metrics[architecture]["test_time"] = test_time
            else:
                performance_metrics[architecture] = {"test_time": test_time}

            print(f"Testing completed in {str(timedelta(seconds=int(test_time)))}")

        except Exception as e:
            print(f"\nWarning: Testing for {architecture} failed with error: {e}")
            traceback.print_exc()
            print(f"Your {architecture} model is still saved at {model_path}")

        # Calculate total time for this architecture
        arch_time = time.time() - arch_start_time
        if architecture in performance_metrics:
            performance_metrics[architecture]["total_time"] = arch_time
        else:
            performance_metrics[architecture] = {"total_time": arch_time}

        print(
            f"Total time for {architecture}: {str(timedelta(seconds=int(arch_time)))}"
        )

        # Write individual architecture metrics to separate file
        try:
            arch_metrics_path = os.path.join(
                metrics_dir, f"{architecture}_performance.json"
            )
            with open(arch_metrics_path, "w") as f:
                # Convert any non-serializable values
                arch_metrics = {}
                for key, value in performance_metrics[architecture].items():
                    if isinstance(value, (torch.Tensor,)):
                        arch_metrics[key] = value.tolist()
                    elif isinstance(value, timedelta):
                        arch_metrics[key] = str(value)
                    else:
                        arch_metrics[key] = value
                json.dump(arch_metrics, f, indent=4)
            print(f"Saved architecture performance metrics to {arch_metrics_path}")
        except Exception as e:
            print(f"Error saving architecture metrics: {e}")

    # Calculate total time for all architectures
    total_time = time.time() - total_start_time

    # Print performance comparison
    if performance_metrics:
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)

        if MODEL_TYPE == "simple":
            print(
                f"{'Architecture':<12} | {'CER':<10} | {'Word Acc':<10} | {'MAD':<8} | {'RMSE':<8} | {'MAPE':<8} | {'Train Time':<15}"
            )
            print("-" * 80)

            for arch, metrics in performance_metrics.items():
                cer = metrics.get(
                    "character_error_rate", metrics.get("final_cer", "N/A")
                )
                if isinstance(cer, float):
                    cer = f"{cer:.2f}%"

                word_acc = metrics.get("word_accuracy", "N/A")
                if isinstance(word_acc, float):
                    word_acc = f"{word_acc:.2f}%"

                mad = metrics.get("MAD", "N/A")
                if isinstance(mad, float):
                    mad = f"{mad:.4f}"

                rmse = metrics.get("RMSE", "N/A")
                if isinstance(rmse, float):
                    rmse = f"{rmse:.4f}"

                mape = metrics.get("MAPE", "N/A")
                if isinstance(mape, float):
                    mape = f"{mape:.2f}%"

                train_time = metrics.get("train_time", "N/A")
                if isinstance(train_time, (int, float)):
                    train_time = str(timedelta(seconds=int(train_time)))

                print(
                    f"{arch:<12} | {cer:<10} | {word_acc:<10} | {mad:<8} | {rmse:<8} | {mape:<8} | {train_time:<15}"
                )

        else:  # hierarchical
            print(
                f"{'Architecture':<18} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'Train Time':<15}"
            )
            print("-" * 80)

            for arch, metrics in performance_metrics.items():
                f1 = metrics.get("f1_score", metrics.get("final_f1", "N/A"))
                if isinstance(f1, float):
                    f1 = f"{f1:.2f}%"

                precision = metrics.get("precision", "N/A")
                if isinstance(precision, float):
                    precision = f"{precision:.2f}%"

                recall = metrics.get("recall", "N/A")
                if isinstance(recall, float):
                    recall = f"{recall:.2f}%"

                train_time = metrics.get("train_time", "N/A")
                if isinstance(train_time, (int, float)):
                    train_time = str(timedelta(seconds=int(train_time)))

                print(
                    f"{arch:<18} | {f1:<10} | {precision:<10} | {recall:<10} | {train_time:<15}"
                )

        # Save performance metrics to a JSON file
        try:
            # Convert timedelta objects to strings for JSON serialization
            serializable_metrics = {}
            for arch, metrics in performance_metrics.items():
                serializable_metrics[arch] = {}
                for key, value in metrics.items():
                    if isinstance(value, timedelta):
                        serializable_metrics[arch][key] = str(value)
                    else:
                        serializable_metrics[arch][key] = value

            metrics_path = os.path.join(base_results_dir, "performance_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(serializable_metrics, f, indent=4)
            print(f"Performance metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
            traceback.print_exc()

    print("\nTotal execution time: " + str(timedelta(seconds=int(total_time))))
    print(
        f"\n{MODEL_TYPE.upper()} OCR system using real multilingual corpus data from OpenSubtitles2016"
    )


if __name__ == "__main__":
    main(should_train=True, path=None)
