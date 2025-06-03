"""
Visualization utilities for OCR system
"""

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def visualize_predictions(
    actual_texts, predicted_texts, languages, architecture, output_path=None
):
    """
    Visualize the comparison between actual and predicted texts
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    plt.suptitle(f"OCR Results Analysis - {architecture.upper()} Model", fontsize=20)

    # 1. Character-level accuracy visualization
    plt.subplot(2, 2, 1)
    plt.title("Character-Level Prediction Accuracy", fontsize=16)

    # Find the longest string
    max_string_length = max(
        max((len(text) for text in actual_texts), default=0),
        max((len(text) for text in predicted_texts), default=0),
    )

    # Prepare data for character-level comparison
    sample_idx = min(10, len(actual_texts))
    char_accuracy = []
    sample_labels = []

    for i in range(sample_idx):
        actual = actual_texts[i]
        pred = predicted_texts[i]

        # Initialize a fixed-length row with padding value (0.5)
        accuracy_row = [0.5] * max_string_length

        # Fill in the actual comparison values
        for j in range(min(len(actual), max_string_length)):
            if j < len(pred):
                # 1 for match, 0 for mismatch
                accuracy_row[j] = 1 if actual[j] == pred[j] else 0
            else:
                # Handle case where pred is shorter than actual
                accuracy_row[j] = 0  # Count as mismatch

        char_accuracy.append(accuracy_row)
        sample_labels.append(f"Sample {i + 1} ({languages[i]})")

    char_data = np.array(char_accuracy)

    # Create heatmap for character-level accuracy
    if char_data.size > 0:
        from matplotlib.colors import ListedColormap
        try:
            import seaborn as sns
            ax = sns.heatmap(
                char_data,
                cmap=ListedColormap(["red", "gray", "green"]),
                vmin=0,
                vmax=1,
                cbar_kws={
                    "ticks": [0.25, 0.5, 0.75],
                    "label": "Character match",
                    "orientation": "horizontal",
                },
                yticklabels=sample_labels,
            )

            # Fix tick positions and labels
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(i) + 1}" for i in ticks], rotation=0)

            plt.xlabel("Character Position")
            plt.ylabel("Sample")
            plt.ylim(len(sample_labels), 0)  # Reverse y-axis to have first sample at top

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="green", label="Match"),
                Patch(facecolor="red", label="Mismatch"),
                Patch(facecolor="gray", label="Padding"),
            ]
            plt.legend(handles=legend_elements, loc="upper right")
        except ImportError:
            # Fallback if seaborn is not available
            plt.imshow(char_data, cmap=ListedColormap(["red", "gray", "green"]), aspect='auto')
            plt.colorbar(label="Character match")
            plt.xlabel("Character Position")
            plt.ylabel("Sample")

    # 2. Per-language accuracy statistics
    plt.subplot(2, 2, 2)
    plt.title("Per-Language Character Accuracy", fontsize=16)

    language_stats = {
        "english": {"correct": 0, "total": 0},
        "romanian": {"correct": 0, "total": 0},
        "hungarian": {"correct": 0, "total": 0},
    }

    # Calculate accuracy per language
    for i in range(len(actual_texts)):
        actual = actual_texts[i]
        pred = predicted_texts[i]
        lang = languages[i]

        # Count correct characters
        for j in range(min(len(actual), len(pred))):
            if actual[j] == pred[j]:
                language_stats[lang]["correct"] += 1
            language_stats[lang]["total"] += 1

        # Count missing or extra characters
        language_stats[lang]["total"] += abs(len(actual) - len(pred))

    # Calculate accuracy percentages
    languages_list = []
    accuracy_values = []

    for lang, stats in language_stats.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            languages_list.append(lang.capitalize())
            accuracy_values.append(accuracy)

    # Sort by accuracy
    if accuracy_values:
        sorted_indices = np.argsort(accuracy_values)
        languages_list = [languages_list[i] for i in sorted_indices]
        accuracy_values = [accuracy_values[i] for i in sorted_indices]

        # Create bar chart for language accuracy
        bars = plt.bar(languages_list, accuracy_values, color="skyblue")
        plt.ylim(0, 100)
        plt.xlabel("Language")
        plt.ylabel("Character Accuracy (%)")

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    # 3. Common errors visualization
    plt.subplot(2, 2, 3)
    plt.title("Most Common Character Prediction Errors", fontsize=16)

    # Collect character-level errors
    errors = []
    for i in range(len(actual_texts)):
        actual = actual_texts[i]
        pred = predicted_texts[i]

        for j in range(min(len(actual), len(pred))):
            if actual[j] != pred[j]:
                errors.append((actual[j], pred[j]))

    # Count error occurrences
    error_counts = Counter(errors)
    most_common_errors = error_counts.most_common(10)  # Get top 10 errors

    if most_common_errors:
        error_labels = [f"'{a}' → '{p}'" for (a, p), _ in most_common_errors]
        error_values = [count for _, count in most_common_errors]

        # Create horizontal bar chart for common errors
        plt.barh(range(len(error_labels)), error_values, color="salmon")
        plt.yticks(range(len(error_labels)), error_labels)
        plt.xlabel("Frequency")
        plt.ylabel("Character Error")
        plt.gca().invert_yaxis()  # Most common at top
    else:
        plt.text(0.5, 0.5, "No errors found", ha="center", va="center")

    # 4. Overall performance metrics
    plt.subplot(2, 2, 4)
    plt.title("Overall Performance Metrics", fontsize=16)
    plt.axis("off")  # Turn off axes

    # Calculate overall metrics
    total_chars = sum(len(text) for text in actual_texts)
    correct_chars = sum(
        sum(1 for a, p in zip(actual, pred) if a == p)
        for actual, pred in zip(actual_texts, predicted_texts)
    )

    if total_chars > 0:
        char_accuracy = (correct_chars / total_chars) * 100
    else:
        char_accuracy = 0

    # Calculate exact matches (perfect predictions)
    exact_matches = sum(
        1 for actual, pred in zip(actual_texts, predicted_texts) if actual == pred
    )
    exact_match_rate = (exact_matches / len(actual_texts)) * 100 if actual_texts else 0

    # Calculate length accuracy
    length_diff = sum(
        abs(len(actual) - len(pred))
        for actual, pred in zip(actual_texts, predicted_texts)
    )
    avg_length_diff = length_diff / len(actual_texts) if actual_texts else 0

    # Display metrics as text
    metrics_text = (
        f"Overall Character Accuracy: {char_accuracy:.2f}%\n\n"
        f"Exactly Matching Samples: {exact_matches}/{len(actual_texts)} ({exact_match_rate:.2f}%)\n\n"
        f"Average Length Difference: {avg_length_diff:.2f} characters\n\n"
        f"Total Samples: {len(actual_texts)}\n\n"
        f"Total Characters: {total_chars}"
    )

    plt.text(0.5, 0.5, metrics_text, ha="center", va="center", fontsize=12)

    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Use the provided output path or default
    if output_path:
        save_path = output_path
    else:
        save_path = f"prediction_analysis_{architecture}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Visualization saved as {save_path}")


def test_and_visualize(
    trainer, data_gen, architecture, num_samples=10, output_dir=None
):
    """
    Test the model on samples from each language and visualize the results
    """
    # Determine output directories
    if output_dir:
        samples_dir = os.path.join(output_dir, "samples")
        plots_dir = os.path.join(output_dir, "plots")
    else:
        samples_dir = "."
        plots_dir = "."

    # Ensure directories exist
    for directory in [samples_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Store results for visualization
    all_actual_texts = []
    all_predicted_texts = []
    all_languages = []

    # Store the original language setting
    original_language = data_gen.language

    # Test samples in each language
    for language in ["english", "romanian", "hungarian"]:
        print(f"\n{language.capitalize()} samples:")

        # Temporarily change the data generator's language
        data_gen.language = language

        # Generate samples specifically in this language
        test_samples = data_gen.generate_batch(batch_size=num_samples, method="corpus")

        for i, sample in enumerate(test_samples):
            # Handle both simple and hierarchical sample formats
            if len(sample) == 3:  # Simple: (text, image, lang)
                text, image, lang = sample
            else:  # Hierarchical: (text, image, lang, multilevel_targets)
                text, image, lang, _ = sample
            
            predicted_text = trainer.predict(image)

            # Print only the first few samples to avoid cluttering the console
            if i < 3:
                print(f"  Sample {i + 1}:")
                print(f"    Actual: '{text}'")
                print(f"    Predicted: '{predicted_text}'")
                print(f"    Language: {lang}")

                # Save sample image
                sample_path = os.path.join(
                    samples_dir, f"{language}_sample_{i + 1}_{architecture}.png"
                )
                image.save(sample_path)
                print(f"Saved sample image to {sample_path}")

            # Store for visualization
            all_actual_texts.append(text)
            all_predicted_texts.append(predicted_text)
            all_languages.append(lang)

    # Restore the original language setting
    data_gen.language = original_language

    # Generate visualizations
    print(f"\nGenerating prediction visualizations for {architecture}...")

    try:
        # Update visualize_predictions call to save to the correct directory
        visualization_path = os.path.join(
            plots_dir, f"prediction_analysis_{architecture}.png"
        )

        # Call to the updated visualize_predictions function
        visualize_predictions(
            all_actual_texts,
            all_predicted_texts,
            all_languages,
            architecture,
            output_path=visualization_path,
        )
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Print the full traceback for debugging
        traceback.print_exc()