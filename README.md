# Quest OCR

A multilingual OCR system supporting English, Romanian, and Hungarian with both simple and hierarchical deep learning models.

## Features

- **Two Model Types:**
  - **Simple**: Fast character-level OCR
  - **Hierarchical**: Multi-level text understanding (characters, n-grams, words)

- **Multiple Architectures:** RNN, GRU, LSTM, Transformer

- **Multilingual Support:** English, Romanian, Hungarian

- **Real Training Data:** Uses OpenSubtitles2016 corpus

## File Structure

```
quest-ocr/
├── main.py                    # Training script (change MODEL_TYPE here)
├── infer.py                   # Inference script
├── config.py                  # Configuration and constants
├── models/                    # Neural network models
├── trainers/                  # Training logic
├── datasets/                  # Data loading and generation
├── utils/                     # Visualization and evaluation
├── corpus_utils.py           # Corpus downloading
├── font_utils.py            # Font handling
└── data_generator.py        # Base data generation
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model:**
   ```bash
   python main.py
   ```

3. **Use for inference:**
   ```python
   from infer import predict_text
   text = predict_text("results_*/*/models/*.pth", "your_image.png")
   ```

## Model Switching

Edit `main.py` and change:
```python
MODEL_TYPE = "simple"       # Fast, good accuracy
# or
MODEL_TYPE = "hierarchical" # Slower, higher accuracy
```
