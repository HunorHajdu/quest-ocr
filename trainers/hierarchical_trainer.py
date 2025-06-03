"""
Hierarchical OCR Trainer
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import timedelta
from sklearn.metrics import precision_score, recall_score, f1_score
from config import IDX_TO_CHAR


class HierarchicalOCRTrainer:
    def __init__(
        self,
        model,
        max_length=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Trainer for hierarchical OCR model without language classification

        Args:
            model: PyTorch OCR model
            max_length: Maximum text length
            device: Device to use for training
        """
        self.model = model
        self.max_length = max_length
        self.device = device
        self.model.to(device)

        # Loss and optimizer
        self.char_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.ngram_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.word_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.optimizer = optim.Adam(model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Statistics per language
        self.language_stats = {
            "english": {"tp": 0, "fp": 0, "fn": 0},
            "romanian": {"tp": 0, "fp": 0, "fn": 0},
            "hungarian": {"tp": 0, "fp": 0, "fn": 0},
        }

    def train_epoch(self, dataloader):
        """
        Train for one epoch

        Args:
            dataloader: PyTorch dataloader with training data

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0

        for images, targets, _ in dataloader:
            images = images.to(self.device)

            # Move all targets to device
            char_targets = targets["char"].to(self.device)
            char_lengths = targets["char_length"].to(self.device)
            word_targets = targets["word"].to(self.device)
            word_lengths = targets["word_length"].to(self.device)
            ngram_targets = targets["ngram"].to(self.device)
            ngram_lengths = targets["ngram_length"].to(self.device)

            # Forward pass
            outputs = self.model(images)
            batch_size = images.size(0)

            # Character-level loss
            char_output = outputs["char"]
            seq_length = char_output.size(1)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=seq_length,
                dtype=torch.long,
                device=self.device,
            )
            char_loss = self.char_criterion(
                char_output.permute(1, 0, 2),
                char_targets,
                input_lengths,
                char_lengths.squeeze(1),
            )

            # N-gram level loss
            ngram_output = outputs["ngram"]
            ngram_seq_length = ngram_output.size(1)
            ngram_input_lengths = torch.full(
                size=(batch_size,),
                fill_value=ngram_seq_length,
                dtype=torch.long,
                device=self.device,
            )
            ngram_loss = self.ngram_criterion(
                ngram_output.permute(1, 0, 2),
                ngram_targets,
                ngram_input_lengths,
                ngram_lengths.squeeze(1),
            )

            # Word-level loss
            word_output = outputs["word"]
            word_seq_length = word_output.size(1)
            word_input_lengths = torch.full(
                size=(batch_size,),
                fill_value=word_seq_length,
                dtype=torch.long,
                device=self.device,
            )
            word_loss = self.word_criterion(
                word_output.permute(1, 0, 2),
                word_targets,
                word_input_lengths,
                word_lengths.squeeze(1),
            )

            # Combined loss with weighting
            # Character-level has highest weight because it's most important
            loss = char_loss * 0.6 + ngram_loss * 0.2 + word_loss * 0.2

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5.0
            )  # Gradient clipping
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluate model

        Args:
            dataloader: PyTorch dataloader with validation data

        Returns:
            Tuple of (average loss, metrics)
        """
        self.model.eval()
        total_loss = 0

        # Metrics collection - use lists to collect results
        all_char_preds = []
        all_char_targets = []

        # Reset language stats
        for lang in self.language_stats:
            self.language_stats[lang]["tp"] = 0
            self.language_stats[lang]["fp"] = 0
            self.language_stats[lang]["fn"] = 0

        with torch.no_grad():
            for images, targets, languages in dataloader:
                images = images.to(self.device)

                # Move all targets to device
                char_targets = targets["char"].to(self.device)
                char_lengths = targets["char_length"].to(self.device)
                word_targets = targets["word"].to(self.device)
                word_lengths = targets["word_length"].to(self.device)
                ngram_targets = targets["ngram"].to(self.device)
                ngram_lengths = targets["ngram_length"].to(self.device)

                # Forward pass
                outputs = self.model(images)
                batch_size = images.size(0)

                # Character-level loss
                char_output = outputs["char"]
                seq_length = char_output.size(1)
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=seq_length,
                    dtype=torch.long,
                    device=self.device,
                )
                char_loss = self.char_criterion(
                    char_output.permute(1, 0, 2),
                    char_targets,
                    input_lengths,
                    char_lengths.squeeze(1),
                )

                # N-gram level loss
                ngram_output = outputs["ngram"]
                ngram_seq_length = ngram_output.size(1)
                ngram_input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=ngram_seq_length,
                    dtype=torch.long,
                    device=self.device,
                )
                ngram_loss = self.ngram_criterion(
                    ngram_output.permute(1, 0, 2),
                    ngram_targets,
                    ngram_input_lengths,
                    ngram_lengths.squeeze(1),
                )

                # Word-level loss
                word_output = outputs["word"]
                word_seq_length = word_output.size(1)
                word_input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=word_seq_length,
                    dtype=torch.long,
                    device=self.device,
                )
                word_loss = self.word_criterion(
                    word_output.permute(1, 0, 2),
                    word_targets,
                    word_input_lengths,
                    word_lengths.squeeze(1),
                )

                # Combined loss
                loss = char_loss * 0.6 + ngram_loss * 0.2 + word_loss * 0.2
                total_loss += loss.item()

                # Decode predictions for evaluation
                decoded_chars = self.decode_predictions(char_output)

                # Collect true and predicted values for metrics
                for i, length in enumerate(char_lengths):
                    target_text = "".join(
                        [IDX_TO_CHAR[idx.item()] for idx in char_targets[i][:length]]
                    )
                    pred_text = decoded_chars[i]
                    language = languages[i]

                    # Store for character-level metrics
                    all_char_targets.extend(list(target_text))
                    all_char_preds.extend(
                        list(pred_text)
                        + ["*"] * max(0, len(target_text) - len(pred_text))
                    )

                    # Character-level TP, FP, FN per language
                    # For each character position, check if prediction matches target
                    for j in range(max(len(target_text), len(pred_text))):
                        if j < len(target_text) and j < len(pred_text):
                            if pred_text[j] == target_text[j]:
                                self.language_stats[language]["tp"] += 1
                            else:
                                self.language_stats[language]["fp"] += 1
                                self.language_stats[language]["fn"] += 1
                        elif j < len(target_text):  # Missing prediction
                            self.language_stats[language]["fn"] += 1
                        else:  # Extra prediction
                            self.language_stats[language]["fp"] += 1

        # Calculate metrics
        metrics = {}

        # Overall precision, recall, f1 at character level
        if all_char_preds and all_char_targets:
            # Binary encoding for precision/recall calculation
            # 1 if prediction matches target, 0 otherwise
            binary_preds = []
            binary_targets = []

            for p, t in zip(all_char_preds, all_char_targets):
                if p == t:
                    binary_preds.append(1)
                    binary_targets.append(1)
                else:
                    binary_preds.append(0)
                    binary_targets.append(1)

            # Calculate metrics
            precision = precision_score(binary_targets, binary_preds, zero_division=0)
            recall = recall_score(binary_targets, binary_preds, zero_division=0)
            f1 = f1_score(binary_targets, binary_preds, zero_division=0)

            metrics["overall_precision"] = precision
            metrics["overall_recall"] = recall
            metrics["overall_f1"] = f1
        else:
            metrics["overall_precision"] = 0
            metrics["overall_recall"] = 0
            metrics["overall_f1"] = 0

        # Language-specific metrics
        for lang, stats in self.language_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            # Calculate precision, recall, f1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[f"{lang}_precision"] = precision
            metrics[f"{lang}_recall"] = recall
            metrics[f"{lang}_f1"] = f1

        return total_loss / len(dataloader), metrics

    def decode_predictions(self, outputs):
        """
        Decode model predictions to text

        Args:
            outputs: Model outputs (log probabilities)

        Returns:
            List of decoded texts
        """
        # Get the most likely character at each step
        _, indices = torch.max(outputs, dim=2)
        indices = indices.cpu().numpy()

        # Convert indices to characters
        decoded_texts = []
        for sequence in indices:
            # Remove duplicates (CTC decoding)
            collapsed = []
            prev_idx = -1
            for idx in sequence:
                if idx != prev_idx and idx != 0:  # Skip if same as previous or padding
                    collapsed.append(idx)
                prev_idx = idx

            # Convert to characters
            text = "".join([IDX_TO_CHAR[idx] for idx in collapsed])
            decoded_texts.append(text)

        return decoded_texts

    def train(self, train_dataloader, val_dataloader, epochs=10):
        """
        Train the model with timing information

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            epochs: Number of epochs

        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "overall_f1": [],
            "english_f1": [],
            "romanian_f1": [],
            "hungarian_f1": [],
            "epoch_times": [],
        }

        # Start timing the entire training process
        training_start_time = time.time()

        for epoch in range(epochs):
            # Start timing this epoch
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_dataloader)
            history["train_loss"].append(train_loss)

            # Evaluate
            val_loss, metrics = self.evaluate(val_dataloader)
            history["val_loss"].append(val_loss)
            history["overall_f1"].append(metrics["overall_f1"])
            history["english_f1"].append(metrics.get("english_f1", 0))
            history["romanian_f1"].append(metrics.get("romanian_f1", 0))
            history["hungarian_f1"].append(metrics.get("hungarian_f1", 0))

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            history["epoch_times"].append(epoch_time)

            # Calculate average epoch time and estimate remaining time
            avg_epoch_time = sum(history["epoch_times"]) / len(history["epoch_times"])
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs

            # Format times for display
            epoch_time_str = str(timedelta(seconds=int(epoch_time)))
            avg_epoch_time_str = str(timedelta(seconds=int(avg_epoch_time)))
            remaining_time_str = str(timedelta(seconds=int(estimated_remaining_time)))

            # Calculate elapsed time
            elapsed_time = epoch_end_time - training_start_time
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))

            # Calculate estimated total time
            estimated_total_time = elapsed_time + estimated_remaining_time
            estimated_total_str = str(timedelta(seconds=int(estimated_total_time)))

            print(
                f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            print(f"  Overall F1: {metrics['overall_f1']:.4f}")
            print(f"  English F1: {metrics.get('english_f1', 0):.4f}")
            print(f"  Romanian F1: {metrics.get('romanian_f1', 0):.4f}")
            print(f"  Hungarian F1: {metrics.get('hungarian_f1', 0):.4f}")
            print(f"  Time for epoch: {epoch_time_str}")
            print(f"  Average epoch time: {avg_epoch_time_str}")
            print(f"  Elapsed time: {elapsed_time_str}")
            print(f"  Estimated remaining time: {remaining_time_str}")
            print(f"  Estimated total training time: {estimated_total_str}")

            # Early stopping: stop if overall F1 > 0.9 (90% accuracy)
            if metrics["overall_f1"] > 0.9:
                print(
                    f"Reached target F1 of {metrics['overall_f1']:.4f} at epoch {epoch + 1}. Stopping early."
                )
                break

        # Calculate actual total training time
        total_training_time = time.time() - training_start_time
        total_time_str = str(timedelta(seconds=int(total_training_time)))
        print(f"\nTotal training time: {total_time_str}")

        return history

    def predict(self, image):
        """
        Predict text from an image

        Args:
            image: PIL Image

        Returns:
            Predicted text
        """
        # Prepare image
        transform = transforms.Compose(
            [
                transforms.Resize((128, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        image = transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)

            # Decode character predictions
            char_pred = self.decode_predictions(outputs["char"])[0]

            # Return predicted text only (no language prediction)
            return char_pred