# Local Training Setup Guide

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv manipulation_env
source manipulation_env/bin/activate  # On Windows: manipulation_env\Scripts\activate

# Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

### 2. Basic Training (Synthetic Data)

```bash
# Train with synthetic data (good for testing)
python local_training_guide.py --use-synthetic --epochs 5 --batch-size 16
```

### 3. Training with Your Own Data

```bash
# Train with CSV data
python local_training_guide.py --data-path your_data.csv --epochs 10 --batch-size 16

# Train with JSON data
python local_training_guide.py --data-path your_data.json --epochs 10 --batch-size 32
```

## 📊 Data Format Requirements

### CSV Format

```csv
text,label
"I think this is reasonable","ethical_persuasion"
"You're being too sensitive","gaslighting"
"If you really cared about me","guilt_tripping"
```

### JSON Format

```json
[
  { "text": "I think this is reasonable", "label": "ethical_persuasion" },
  { "text": "You're being too sensitive", "label": "gaslighting" },
  { "text": "If you really cared about me", "label": "guilt_tripping" }
]
```

## ⚙️ Training Configuration

### Basic Configuration

```bash
python local_training_guide.py \
  --data-path data.csv \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 512 \
  --output-dir ./models
```

### Advanced Configuration

```bash
python local_training_guide.py \
  --data-path large_dataset.csv \
  --model-name "roberta-base" \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --max-length 256 \
  --output-dir ./trained_models
```

## 🎯 Model Options

### Recommended Models by Use Case

**Fast Training (CPU/Small GPU):**

- `distilbert-base-uncased` (66M parameters)
- `albert-base-v2` (12M parameters)

**Best Performance (Large GPU):**

- `roberta-base` (125M parameters)
- `bert-base-uncased` (110M parameters)

**Multilingual:**

- `distilbert-base-multilingual-cased`
- `xlm-roberta-base`

## 💾 Hardware Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 2GB free space
- **Training Time**: 2-4 hours (CPU, synthetic data)

### Recommended Requirements

- **GPU**: NVIDIA GTX 1060+ (6GB VRAM)
- **RAM**: 16GB+
- **Storage**: 5GB free space
- **Training Time**: 30-60 minutes (GPU, real data)

### Optimal Requirements

- **GPU**: NVIDIA RTX 3080+ (10GB+ VRAM)
- **RAM**: 32GB+
- **Storage**: 10GB free space
- **Training Time**: 10-20 minutes (GPU, large dataset)

## 📈 Training Process

### What Happens During Training

1. **Data Loading**: Loads and preprocesses your text data
2. **Model Setup**: Initializes transformer model with classification head
3. **Training Loop**:
   - Forward pass through model
   - Calculate loss and gradients
   - Update model weights
   - Validate on held-out data
4. **Model Saving**: Saves best model based on validation accuracy
5. **Visualization**: Generates training curves and metrics

### Expected Output

```
INFO - Using device: cuda
INFO - Loaded 2200 examples from data.csv
INFO - Training samples: 1760
INFO - Validation samples: 440
INFO - Model parameters: 66,371,339

Epoch 1/10
Training: 100%|██████████| 110/110 [02:15<00:00]
Validation: 100%|██████████| 28/28 [00:15<00:00]
INFO - Train Loss: 2.1234, Train Acc: 0.3456
INFO - Val Loss: 1.8765, Val Acc: 0.4321
INFO - New best model saved: best_model_epoch_1.pt

...

INFO - Training completed! Best validation accuracy: 0.8765
INFO - Final model saved to: ./models/final_model.pt
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Error:**

```bash
# Reduce batch size
python local_training_guide.py --batch-size 8

# Reduce sequence length
python local_training_guide.py --max-length 256

# Use smaller model
python local_training_guide.py --model-name "distilbert-base-uncased"
```

**Slow Training:**

```bash
# Check GPU usage
nvidia-smi

# Use mixed precision (if supported)
# Add to training script: --fp16

# Increase batch size if you have memory
python local_training_guide.py --batch-size 32
```

**Poor Performance:**

```bash
# Increase training epochs
python local_training_guide.py --epochs 20

# Lower learning rate
python local_training_guide.py --learning-rate 1e-5

# Use larger model
python local_training_guide.py --model-name "roberta-base"
```

## 📊 Monitoring Training

### Training Curves

The script automatically generates `training_curves.png` showing:

- Training vs Validation Loss
- Training vs Validation Accuracy

### Log Files

- `training.log` - Detailed training logs
- Console output - Real-time progress

### Model Checkpoints

- `best_model_epoch_X.pt` - Best model during training
- `final_model.pt` - Final model after all epochs

## 🎯 Using Your Trained Model

### Load and Use Trained Model

```python
import torch
from transformers import AutoTokenizer

# Load model
checkpoint = torch.load('models/final_model.pt')
model = LocalManipulationClassifier()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Make predictions
text = "You're being too sensitive about this."
result = model.predict(text, tokenizer, device='cpu')
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Export for Production

```python
# Save for deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer_name': 'distilbert-base-uncased',
    'label_mapping': model.label_to_id,
    'config': config
}, 'production_model.pt')
```

## 📚 Next Steps

### Improve Performance

1. **Collect More Data**: Aim for 1000+ examples per class
2. **Data Quality**: Clean and validate your training data
3. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
4. **Model Architecture**: Try different transformer models
5. **Ensemble Methods**: Combine multiple models

### Deploy Your Model

1. **Export to ONNX**: For cross-platform deployment
2. **Create API**: Use FastAPI or Flask
3. **Containerize**: Use Docker for easy deployment
4. **Monitor Performance**: Track accuracy in production

### Advanced Features

1. **Active Learning**: Iteratively improve with new data
2. **Explainability**: Add attention visualization
3. **Multi-label**: Detect multiple manipulation tactics
4. **Real-time**: Optimize for low-latency inference

## 🆘 Support

If you encounter issues:

1. **Check Requirements**: Ensure all dependencies are installed
2. **Review Logs**: Check `training.log` for detailed error messages
3. **Reduce Complexity**: Start with smaller models/datasets
4. **Hardware Check**: Verify GPU/memory availability
5. **Data Validation**: Ensure your data format is correct

Happy training! 🚀
