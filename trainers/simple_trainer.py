"""
Simple OCR Trainer
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import timedelta
from config import IDX_TO_CHAR


class OCRTrainer:
    def __init__(
        self,
        model,
        max_length=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Trainer for OCR model

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
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Statistics per language
        self.language_stats = {
            "english": {"total_correct": 0, "total_chars": 0},
            "romanian": {"total_correct": 0, "total_chars": 0},
            "hungarian": {"total_correct": 0, "total_chars": 0},
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

        for images, targets, target_lengths, _ in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass
            outputs = self.model(images)
            batch_size, sequence_length, _ = outputs.size()

            # Prepare for CTC loss
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=sequence_length,
                dtype=torch.long,
                device=self.device,
            )

            # Calculate loss
            loss = self.criterion(
                outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
            )

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluate model

        Args:
            dataloader: PyTorch dataloader with validation data

        Returns:
            Tuple of (average loss, character error rate, per-language character error rates)
        """
        self.model.eval()
        total_loss = 0

        # Reset language stats
        for lang in self.language_stats:
            self.language_stats[lang]["total_correct"] = 0
            self.language_stats[lang]["total_chars"] = 0

        with torch.no_grad():
            for images, targets, target_lengths, languages in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward pass
                outputs = self.model(images)
                batch_size, sequence_length, _ = outputs.size()

                # Prepare for CTC loss
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=sequence_length,
                    dtype=torch.long,
                    device=self.device,
                )

                # Calculate loss
                loss = self.criterion(
                    outputs.permute(1, 0, 2), targets, input_lengths, target_lengths
                )
                total_loss += loss.item()

                # Decode predictions
                decoded_preds = self.decode_predictions(outputs)

                # Calculate accuracy per language
                for i, length in enumerate(target_lengths):
                    target_text = "".join(
                        [IDX_TO_CHAR[idx.item()] for idx in targets[i][:length]]
                    )
                    pred_text = decoded_preds[i]
                    language = languages[i]

                    # Update language-specific stats
                    for p_char, t_char in zip(pred_text, target_text):
                        if p_char == t_char:
                            self.language_stats[language]["total_correct"] += 1
                    self.language_stats[language]["total_chars"] += len(target_text)

        # Calculate overall and per-language CER
        total_correct = sum(
            lang_stat["total_correct"] for lang_stat in self.language_stats.values()
        )
        total_chars = sum(
            lang_stat["total_chars"] for lang_stat in self.language_stats.values()
        )

        overall_cer = 1 - (total_correct / total_chars) if total_chars > 0 else 1.0

        language_cer = {}
        for lang, stats in self.language_stats.items():
            if stats["total_chars"] > 0:
                language_cer[lang] = 1 - (stats["total_correct"] / stats["total_chars"])
            else:
                language_cer[lang] = 1.0

        return total_loss / len(dataloader), overall_cer, language_cer

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
            "overall_cer": [],
            "english_cer": [],
            "romanian_cer": [],
            "hungarian_cer": [],
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
            val_loss, overall_cer, language_cer = self.evaluate(val_dataloader)
            history["val_loss"].append(val_loss)
            history["overall_cer"].append(overall_cer)
            history["english_cer"].append(language_cer.get("english", 1.0))
            history["romanian_cer"].append(language_cer.get("romanian", 1.0))
            history["hungarian_cer"].append(language_cer.get("hungarian", 1.0))

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
            print(f"  Overall CER: {overall_cer:.4f}")
            print(f"  English CER: {language_cer.get('english', 1.0):.4f}")
            print(f"  Romanian CER: {language_cer.get('romanian', 1.0):.4f}")
            print(f"  Hungarian CER: {language_cer.get('hungarian', 1.0):.4f}")
            print(f"  Time for epoch: {epoch_time_str}")
            print(f"  Average epoch time: {avg_epoch_time_str}")
            print(f"  Elapsed time: {elapsed_time_str}")
            print(f"  Estimated remaining time: {remaining_time_str}")
            print(f"  Estimated total training time: {estimated_total_str}")

            # Early stopping: stop if overall CER < 0.1 (90% accuracy)
            if overall_cer < 0.1:
                print(
                    f"Reached target CER of {overall_cer:.4f} at epoch {epoch + 1}. Stopping early."
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

        # Decode prediction
        decoded_text = self.decode_predictions(outputs)[0]
        return decoded_text