"""
Pneumonia Detection Source Package
Contains dataset, model, training, inference, and explainability modules.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .dataset import ChestXRayDataset, get_dataloaders, get_transforms
from .model import ResNet18Pneumonia, EfficientNetPneumonia, get_model
from .train import PneumoniaTrainer, train_model
from .inference import PneumoniaPredictor, load_and_predict
from .gradcam import GradCAM, create_gradcam_explainer
from .utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    compute_metrics,
    plot_metrics,
    plot_confusion_matrix,
    EarlyStopping
)

__all__ = [
    # Dataset
    'ChestXRayDataset',
    'get_dataloaders',
    'get_transforms',
    
    # Models
    'ResNet18Pneumonia',
    'EfficientNetPneumonia',
    'get_model',
    
    # Training
    'PneumoniaTrainer',
    'train_model',
    
    # Inference
    'PneumoniaPredictor',
    'load_and_predict',
    
    # Explainability
    'GradCAM',
    'create_gradcam_explainer',
    
    # Utilities
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'compute_metrics',
    'plot_metrics',
    'plot_confusion_matrix',
    'EarlyStopping',
]
