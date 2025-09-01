# 🚀 Kaggle Enhanced Splits Training Guide

## Complete Step-by-Step Instructions for GPU Training

### 📋 **Prerequisites**

- Kaggle account
- Enhanced_critical_splits.json dataset
- Basic familiarity with Kaggle notebooks

---

## 🎯 **Step 1: Prepare Your Dataset**

### Upload Dataset to Kaggle

1. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets
2. **Click "New Dataset"**
3. **Upload your `enhanced_critical_splits.json` file**
4. **Set dataset details**:
   - Title: "Manipulation Detection Enhanced Splits"
   - Description: "Enhanced critical splits dataset for manipulation detection training"
   - Visibility: Private (recommended)
5. **Click "Create Dataset"**
6. **Note the dataset URL** (you'll need this later)

---

## 🎯 **Step 2: Create Kaggle Notebook**

### Setup New Notebook

1. **Go to Kaggle Notebooks**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Choose "Notebook" (not Script)**
4. **Enable GPU**:
   - Click "Settings" (gear icon on right)
   - Under "Accelerator", select **"GPU P100"** or **"GPU T4 x2"**
   - Click "Save"

### Add Your Dataset

1. **In notebook, click "Add Data"** (+ icon on right)
2. **Search for your dataset** by name
3. **Click "Add"** to attach it to notebook
4. **Note the path**: `/kaggle/input/psychological-manipulation-detection-dataset/enhanced_critical_splits.json`

---

## 🎯 **Step 3: Install Dependencies**

**Copy and run this in your first notebook cell:**

```python
# Install required packages
!pip install transformers torch torchvision torchaudio --quiet
!pip install scikit-learn pandas numpy matplotlib seaborn tqdm --quiet

print("✅ Dependencies installed successfully!")
```

---

## 🎯 **Step 4: Copy Training Code**

**Copy the ENTIRE contents of `kaggle_enhanced_splits_trainer.py` into a new cell.**

**⚠️ IMPORTANT: Update the dataset path in the code:**

Find this line in the code:

```python
train_data, val_data, test_data = load_enhanced_splits_data(
    '/kaggle/input/psychological-manipulation-detection-dataset/enhanced_critical_splits.json'
)
```

The dataset path is already configured for the `psychological-manipulation-detection-dataset`.

---

## 🎯 **Step 5: Run Training**

### Execute the Training Cell

1. **Run the cell** with the training code
2. **Expected output**:

```
🚀 Kaggle Enhanced Splits Trainer
==================================================
PyTorch version: 1.13.1+cu116
CUDA available: True
GPU: Tesla P100-PCIE-16GB
GPU Memory: 16.3 GB

📂 Loading data from: /kaggle/input/psychological-manipulation-detection-dataset/enhanced_critical_splits.json
✅ Loaded 3617 training samples
✅ Loaded 772 validation samples
✅ Loaded 781 test samples

📊 Training set class distribution:
  appeal_to_emotion        :  175 ( 4.8%)
  belittling_ridicule      :  560 (15.5%)
  deflection               :   61 ( 1.7%)
  ethical_persuasion       : 1260 (34.8%)
  gaslighting              :  560 (15.5%)
  guilt_tripping           :  193 ( 5.3%)
  love_bombing             :  225 ( 6.2%)
  passive_aggression       :   69 ( 1.9%)
  stonewalling             :  175 ( 4.8%)
  threatening_intimidation :  280 ( 7.7%)
  whataboutism             :   59 ( 1.6%)

🎯 Starting training on cuda
📊 Training samples: 3617
📊 Validation samples: 772
⚙️ Epochs: 15
⚙️ Batch size: 32
⚙️ Learning rate: 2e-05
```

### Monitor Training Progress

You'll see progress bars and metrics for each epoch:

```
📈 Epoch 1/15
Training: 100%|██████████| 114/114 [02:15<00:00, loss=2.234, acc=0.456, lr=1.2e-05]
Validation: 100%|██████████| 25/25 [00:15<00:00]
📊 Train Loss: 2.2340, Train Acc: 0.4560
📊 Val Loss: 1.8765, Val Acc: 0.5432
💾 New best model saved! Validation accuracy: 0.5432
```

---

## 🎯 **Step 6: Training Configuration Options**

### Default Configuration (Recommended)

```python
config = {
    'model_name': 'distilbert-base-uncased',  # Fast and efficient
    'epochs': 15,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'max_length': 512,
    'patience': 5  # Early stopping
}
```

### For Faster Training (Lower Quality)

```python
config = {
    'model_name': 'distilbert-base-uncased',
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 3e-5,
    'max_length': 256,
    'patience': 3
}
```

### For Best Performance (Slower)

```python
config = {
    'model_name': 'roberta-base',
    'epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-5,
    'max_length': 512,
    'patience': 7
}
```

---

## 🎯 **Step 7: Expected Results**

### Training Time

- **GPU P100**: ~45-60 minutes for 15 epochs
- **GPU T4**: ~30-45 minutes for 15 epochs
- **CPU**: ~4-6 hours (not recommended)

### Expected Accuracy

- **Target**: 85%+ validation accuracy
- **Good**: 80-85% validation accuracy
- **Acceptable**: 75-80% validation accuracy

### Training Output

```
🎉 Training completed successfully!
🏆 Best validation accuracy: 0.8756

📈 Final Training Metrics:
  Best Validation Accuracy: 0.8756
  Final Training Accuracy: 0.9123
  Final Validation Loss: 0.4321

🧪 Testing model predictions:
==================================================
 1. Text: 'I think this is a reasonable approach to solve our problem...'
    Prediction: Ethical Persuasion
    Confidence: 0.892
    Is Manipulation: No
--------------------------------------------------
 2. Text: 'You're being way too sensitive about this whole situation...'
    Prediction: Gaslighting
    Confidence: 0.834
    Is Manipulation: Yes
```

---

## 🎯 **Step 8: Save and Download Model**

### Files Created

After training, you'll have:

- `best_kaggle_model.pt` - Best model during training
- `final_kaggle_manipulation_model.pt` - Final model with metadata

### Download Models

1. **In Kaggle notebook**: Files appear in output section
2. **Click "Download"** to save to your computer
3. **Use for deployment** or further testing

---

## 🔧 **Troubleshooting**

### Common Issues and Solutions

#### **"Dataset not found" Error**

```python
# Check available datasets
import os
print("Available datasets:")
for item in os.listdir('/kaggle/input/'):
    print(f"  {item}")
```

#### **Out of Memory Error**

```python
# Reduce batch size in config
config['batch_size'] = 16  # or even 8
config['max_length'] = 256  # shorter sequences
```

#### **Slow Training**

```python
# Check GPU usage
!nvidia-smi

# Increase batch size if memory allows
config['batch_size'] = 64
```

#### **Poor Performance**

```python
# Train longer
config['epochs'] = 25

# Use better model
config['model_name'] = 'roberta-base'

# Lower learning rate
config['learning_rate'] = 1e-5
```

---

## 🎯 **Step 9: Using Your Trained Model**

### Load and Test Model

```python
# Load the trained model
checkpoint = torch.load('/kaggle/working/final_kaggle_manipulation_model.pt')

# Extract components
model_state = checkpoint['model_state_dict']
label_mapping = checkpoint['label_mapping']
config = checkpoint['config']

print(f"Model accuracy: {checkpoint['best_val_acc']:.4f}")
print(f"Classes: {list(label_mapping.keys())}")
```

### Make Predictions

```python
# Quick prediction function
def predict_manipulation(text):
    # Load model (you'd do this once in practice)
    model = KaggleManipulationClassifier(num_classes=len(label_mapping))
    model.load_state_dict(model_state)
    model.eval()

    # Tokenize and predict
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()

    # Get label
    id_to_label = {v: k for k, v in label_mapping.items()}
    predicted_label = id_to_label[predicted_id]

    return {
        'text': text,
        'predicted_class': predicted_label,
        'confidence': confidence,
        'is_manipulation': predicted_label != 'ethical_persuasion'
    }

# Test it
result = predict_manipulation("You're being too sensitive about this.")
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

---

## 🎯 **Step 10: Advanced Options**

### Experiment Tracking

```python
# Add to training code for better tracking
import wandb

# Initialize wandb (optional)
wandb.init(project="manipulation-detection", config=config)

# Log metrics during training
wandb.log({"train_loss": train_loss, "val_acc": val_acc})
```

### Model Comparison

```python
# Try different models
models_to_try = [
    'distilbert-base-uncased',
    'roberta-base',
    'bert-base-uncased',
    'albert-base-v2'
]

for model_name in models_to_try:
    config['model_name'] = model_name
    # Run training...
```

### Hyperparameter Tuning

```python
# Grid search over hyperparameters
learning_rates = [1e-5, 2e-5, 3e-5]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        config['learning_rate'] = lr
        config['batch_size'] = bs
        # Run training...
```

---

## 🎉 **Success Checklist**

- [ ] ✅ Dataset uploaded to Kaggle
- [ ] ✅ GPU enabled in notebook
- [ ] ✅ Dependencies installed
- [ ] ✅ Dataset path updated in code
- [ ] ✅ Training completed without errors
- [ ] ✅ Validation accuracy > 80%
- [ ] ✅ Model files downloaded
- [ ] ✅ Test predictions working

---

## 🆘 **Need Help?**

### Quick Debugging

1. **Check GPU**: `torch.cuda.is_available()` should return `True`
2. **Check Data**: Verify dataset path and file structure
3. **Check Memory**: Monitor GPU memory usage with `nvidia-smi`
4. **Check Logs**: Look for error messages in training output

### Performance Tips

- **Use GPU P100 or T4** for best performance
- **Start with default config** before experimenting
- **Monitor validation accuracy** - should improve each epoch
- **Use early stopping** to prevent overfitting

Happy training! 🚀
