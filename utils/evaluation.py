"""
Evaluation utilities for OCR system
"""

import os
import json
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


def calculate_levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def evaluate_model_metrics(
    trainer, data_gen, architecture, metrics_dir, num_samples=100
):
    """
    Evaluate model using various metrics (MAPE, MAD, RMSE, MSE)
    """
    # Create metrics directory if it doesn't exist
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Generate test data
    test_data = data_gen.generate_batch(batch_size=num_samples, method="corpus")

    # Keep track of all prediction results
    char_preds = []  # character-level predictions
    word_preds = []  # word-level predictions

    # Store for calculating metrics
    all_actual_texts = []
    all_predicted_texts = []
    all_languages = []
    all_char_counts = []

    # Evaluate each sample
    for sample in test_data:
        # Handle both simple and hierarchical sample formats
        if len(sample) == 3:  # Simple: (text, image, lang)
            text, image, language = sample
        else:  # Hierarchical: (text, image, lang, multilevel_targets)
            text, image, language, _ = sample
            
        predicted_text = trainer.predict(image)

        all_actual_texts.append(text)
        all_predicted_texts.append(predicted_text)
        all_languages.append(language)
        all_char_counts.append(len(text))

        # Character-level (Levenshtein distance)
        edit_distance = calculate_levenshtein_distance(text, predicted_text)
        char_error_rate = edit_distance / len(text) if len(text) > 0 else 1.0
        char_preds.append((char_error_rate, edit_distance, len(text)))

        # Word-level (exact match)
        is_correct = 1 if text == predicted_text else 0
        word_preds.append(is_correct)

    # Calculate metrics
    metrics = {}

    # Word-level metrics
    word_accuracy = sum(word_preds) / len(word_preds) if word_preds else 0
    metrics["word_accuracy"] = word_accuracy * 100  # as percentage

    # Character-level metrics
    error_rates = [r for r, _, _ in char_preds]

    metrics["character_error_rate"] = np.mean(error_rates) * 100  # as percentage

    # Calculate edit distances for absolute metrics
    edit_distances = [d for _, d, _ in char_preds]
    text_lengths = [l for _, _, l in char_preds]

    # Mean Absolute Error (MAE/MAD)
    metrics["MAD"] = float(
        mean_absolute_error(np.zeros_like(edit_distances), edit_distances)
    )

    # Mean Squared Error (MSE)
    metrics["MSE"] = float(
        mean_squared_error(np.zeros_like(edit_distances), edit_distances)
    )

    # Root Mean Squared Error (RMSE)
    metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))

    # Mean Absolute Percentage Error (MAPE)
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_errors = np.array(
            [d / l * 100 for d, l in zip(edit_distances, text_lengths)]
        )
        # Filter out NaN and inf values
        percentage_errors = percentage_errors[~np.isnan(percentage_errors)]
        percentage_errors = percentage_errors[~np.isinf(percentage_errors)]
        metrics["MAPE"] = float(
            np.mean(percentage_errors) if len(percentage_errors) > 0 else 0
        )

    # Per-language metrics
    for lang in ["english", "romanian", "hungarian"]:
        lang_indices = [i for i, l in enumerate(all_languages) if l == lang]
        if lang_indices:
            lang_errors = [error_rates[i] for i in lang_indices]
            metrics[f"{lang}_error_rate"] = float(
                np.mean(lang_errors) * 100
            )  # as percentage

    # Save metrics to JSON
    metrics_path = os.path.join(metrics_dir, f"{architecture}_metrics.json")

    # Ensure all values are JSON serializable
    for key, value in metrics.items():
        if isinstance(value, np.float32) or isinstance(value, np.float64):
            metrics[key] = float(value)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Log metrics summary
    logger.info(f"\nMetrics for {architecture}:")
    logger.info(f"  Word Accuracy: {metrics['word_accuracy']:.2f}%")
    logger.info(f"  Character Error Rate: {metrics['character_error_rate']:.2f}%")
    logger.info(f"  MAD: {metrics['MAD']:.4f}")
    logger.info(f"  MSE: {metrics['MSE']:.4f}")
    logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")

    return metrics


def evaluate_hierarchical_model_metrics(
    trainer, data_gen, architecture, metrics_dir, num_samples=100
):
    """
    Evaluate hierarchical model using precision, recall, and F1 score
    """
    # Create metrics directory if it doesn't exist
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Generate test data
    test_data = data_gen.generate_batch(batch_size=num_samples, method="corpus")

    # Store for calculating metrics
    language_stats = {
        "english": {"tp": 0, "fp": 0, "fn": 0},
        "romanian": {"tp": 0, "fp": 0, "fn": 0},
        "hungarian": {"tp": 0, "fp": 0, "fn": 0},
    }

    # Evaluate each sample
    for text, image, language, _ in test_data:
        predicted_text = trainer.predict(image)

        # Count true positives, false positives, false negatives
        for j in range(max(len(text), len(predicted_text))):
            if j < len(text) and j < len(predicted_text):
                if predicted_text[j] == text[j]:
                    language_stats[language]["tp"] += 1
                else:
                    language_stats[language]["fp"] += 1
                    language_stats[language]["fn"] += 1
            elif j < len(text):  # Missing prediction
                language_stats[language]["fn"] += 1
            else:  # Extra prediction
                language_stats[language]["fp"] += 1

    # Calculate metrics
    metrics = {}

    # Overall metrics
    all_tp = sum(stats["tp"] for stats in language_stats.values())
    all_fp = sum(stats["fp"] for stats in language_stats.values())
    all_fn = sum(stats["fn"] for stats in language_stats.values())

    metrics["precision"] = (
        all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
    )
    metrics["recall"] = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0
    metrics["f1_score"] = (
        (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"])
        )
        if (metrics["precision"] + metrics["recall"]) > 0
        else 0
    )

    # Per-language metrics
    for lang, stats in language_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[f"{lang}_precision"] = precision
        metrics[f"{lang}_recall"] = recall
        metrics[f"{lang}_f1"] = f1

    # Save metrics to JSON
    metrics_path = os.path.join(metrics_dir, f"{architecture}_metrics.json")

    # Ensure all values are JSON serializable
    for key, value in metrics.items():
        if isinstance(value, np.float32) or isinstance(value, np.float64):
            metrics[key] = float(value)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Log metrics summary
    logger.info(f"\nMetrics for {architecture}:")
    logger.info(f"  Precision: {metrics['precision']:.2f}%")
    logger.info(f"  Recall: {metrics['recall']:.2f}%")
    logger.info(f"  F1 Score: {metrics['f1_score']:.2f}%")

    return metrics