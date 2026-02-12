# 🫁 Explainable Deep Learning System for Pneumonia Detection

A comprehensive end-to-end deep learning project that detects pneumonia from chest X-ray images using transfer learning and provides visual explanations of predictions through Grad-CAM.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Explainability](#explainability)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements an AI-powered pneumonia detection system that:
- Classifies chest X-ray images as **Normal** or **Pneumonia**
- Achieves high accuracy using transfer learning with **ResNet50** or **EfficientNet-B0**
- Provides visual explanations using **Grad-CAM** to highlight regions of interest
- Offers an intuitive **Streamlit web interface** for real-time inference

**⚠️ Disclaimer:** This is an educational project for learning and demonstration purposes. It is NOT intended for clinical use or medical diagnosis.

## ✨ Features

- **Transfer Learning**: Utilizes pre-trained ImageNet models (ResNet50/EfficientNet-B0)
- **Data Augmentation**: Robust preprocessing with random rotations, flips, and color jittering
- **Explainable AI**: Grad-CAM heatmaps show which regions influenced the prediction
- **Web Interface**: User-friendly Streamlit app for image upload and inference
- **Comprehensive Evaluation**: Metrics include accuracy, precision, recall, and F1-score
- **Modular Codebase**: Clean, well-documented, and reusable components

## 📁 Project Structure

```
pneumonia-detection/
├── data/
│   └── chest_xray/              # Dataset directory (not included in repo)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       └── test/
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and visualization
│   ├── 02_preprocessing.ipynb       # Data preprocessing pipeline
│   └── 03_training.ipynb            # Model training notebook
├── src/
│   ├── dataset.py               # Dataset and DataLoader utilities
│   ├── model.py                 # Model architectures (ResNet50, EfficientNet)
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Inference pipeline
│   ├── gradcam.py               # Grad-CAM implementation
│   └── utils.py                 # Helper functions
├── checkpoints/                 # Saved model weights (created during training)
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended but not required)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

### Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download the dataset
3. Extract it into the `data/chest_xray/` directory

The dataset should be organized as:
```
data/chest_xray/
├── train/
│   ├── NORMAL/     (~1,341 images)
│   └── PNEUMONIA/  (~3,875 images)
├── val/
│   ├── NORMAL/     (~8 images)
│   └── PNEUMONIA/  (~8 images)
└── test/
    ├── NORMAL/     (~234 images)
    └── PNEUMONIA/  (~390 images)
```

### Dataset Statistics

- **Total Images**: ~5,856
- **Classes**: Normal, Pneumonia
- **Image Format**: JPEG/PNG
- **Image Size**: Variable (resized to 224x224 for training)
- **Class Distribution**: Imbalanced (~70% Pneumonia)

## 🎮 Usage

### 1. Data Exploration

Explore the dataset using the Jupyter notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Training the Model

Train the model using either the notebook or the training script:

**Option A: Using Jupyter Notebook**
```bash
jupyter notebook notebooks/03_training.ipynb
```

**Option B: Using Python Script**
```bash
python -c "from src.train import train_model; train_model(model_name='resnet50', num_epochs=20)"
```

Training parameters:
- `model_name`: 'resnet50' or 'efficientnet_b0'
- `batch_size`: 32 (default)
- `num_epochs`: 20 (default)
- `learning_rate`: 1e-4 (default)

The trained model will be saved to `checkpoints/best_model.pt`.

### 3. Running the Streamlit App

Launch the web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

**How to use the app:**
1. Upload a chest X-ray image (JPEG/PNG)
2. Click "Analyze Image"
3. View the prediction, confidence score, and Grad-CAM heatmap

### 4. Making Predictions Programmatically

```python
from src.inference import PneumoniaPredictor

# Load model
predictor = PneumoniaPredictor(
    model_path='checkpoints/best_model.pt',
    model_name='resnet50'
)

# Predict
result = predictor.predict('path/to/xray.jpg', return_gradcam=True)

print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🏗️ Model Architecture

### ResNet50 (Default)

- **Base**: Pre-trained ResNet50 on ImageNet
- **Modifications**: 
  - Replaced final FC layer with custom classifier
  - Added dropout (0.5, 0.3) for regularization
  - Custom head: 2048 → 512 → 256 → 2
- **Input Size**: 224x224x3
- **Output**: 2 classes (Normal, Pneumonia)

### EfficientNet-B0 (Alternative)

- **Base**: Pre-trained EfficientNet-B0 on ImageNet
- **Modifications**:
  - Custom classifier with dropout
  - 1280 → 256 → 2
- **Input Size**: 224x224x3
- **Output**: 2 classes

### Training Strategy

1. **Transfer Learning**: Use ImageNet pre-trained weights
2. **Fine-tuning**: Initially freeze backbone, then unfreeze for fine-tuning
3. **Optimizer**: Adam with weight decay (1e-5)
4. **Loss Function**: Cross-Entropy Loss
5. **Learning Rate Scheduler**: ReduceLROnPlateau
6. **Early Stopping**: Patience of 5 epochs

## 📈 Results

Expected performance metrics (after training):

| Metric      | Score  |
|-------------|--------|
| Accuracy    | ~92%   |
| Precision   | ~90%   |
| Recall      | ~95%   |
| F1-Score    | ~92%   |

*Note: Actual results may vary based on training configuration and random seed.*

### Training Curves

Training and validation loss/accuracy curves will be generated during training and saved in the notebooks.

### Confusion Matrix

The model's performance is analyzed using a confusion matrix showing:
- True Positives (Pneumonia correctly identified)
- True Negatives (Normal correctly identified)
- False Positives (Normal misclassified as Pneumonia)
- False Negatives (Pneumonia misclassified as Normal)

## 🔍 Explainability

### Grad-CAM (Gradient-weighted Class Activation Mapping)

The project implements **Grad-CAM** to provide visual explanations of model predictions.

**How it works:**
1. Extracts gradients from the last convolutional layer
2. Computes weighted activation maps
3. Generates a heatmap highlighting important regions
4. Overlays the heatmap on the original image

**Interpretation:**
- **Red/Yellow regions**: High importance for prediction
- **Blue regions**: Low importance

This helps clinicians and researchers understand which parts of the X-ray influenced the model's decision.

## 🔮 Future Improvements

- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Ensemble models for improved accuracy
- [ ] Integration with DICOM format support
- [ ] Deployment on cloud platforms (AWS, GCP, Azure)
- [ ] Mobile app development
- [ ] Real-time video stream analysis
- [ ] Integration with hospital PACS systems
- [ ] Uncertainty quantification for predictions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework
- **Grad-CAM**: [Original paper](https://arxiv.org/abs/1610.02391) by Selvaraju et al.

## 👨‍💻 Author

**Your Name**  
Third-year Computer Science Student  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📞 Contact

For questions, suggestions, or collaboration:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/pneumonia-detection/issues)

---

**⚠️ IMPORTANT MEDICAL DISCLAIMER**

This tool is for **educational and research purposes only**. It is **NOT a medical device** and should **NOT be used for clinical diagnosis or treatment**. Always consult qualified healthcare professionals for medical advice. The developers assume no liability for any consequences arising from the use of this software.

---

*Made with ❤️ for deep learning and healthcare AI*
