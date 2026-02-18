"""
Inference pipeline for predictions on single images.
Handles loading, preprocessing, and prediction with explanations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from src.model import get_model
from src.gradcam import GradCAM
from src.utils import get_device


class PneumoniaPredictor:
    """
    Inference pipeline for pneumonia detection.
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = 'resnet18',
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to saved model checkpoint
            model_name (str): Name of model architecture
            device (torch.device, optional): Device to use
        """
        self.device = device or get_device()
        self.model_name = model_name
        self.class_names = ['Normal', 'Pneumonia']
        
        # Load model
        self.model = get_model(model_name, num_classes=2, pretrained=False)
        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Create GradCAM
        self.gradcam = GradCAM(self.model, self.model.get_last_conv_layer())
        
        # Preprocessing transform (ImageNet normalization)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
    
    def _load_checkpoint(self, model_path: str):
        """
        Load model from checkpoint.
        
        Args:
            model_path (str): Path to checkpoint
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image_path (str): Path to image
            image_size (int): Target image size
            
        Returns:
            Tuple[torch.Tensor, np.ndarray]: Preprocessed tensor and original numpy array
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Store original for visualization
        image_np = np.array(image)
        
        # Resize
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image_np_resized = np.array(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np_resized).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Normalize with ImageNet stats
        image_tensor = (image_tensor - self.mean) / self.std
        
        return image_tensor, image_np_resized
    
    def predict(
        self,
        image_path: str,
        return_gradcam: bool = True
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path (str): Path to image
            return_gradcam (bool): Whether to generate Grad-CAM explanation
            
        Returns:
            Dict: Prediction results including label, confidence, and optionally Grad-CAM
        """
        # Preprocess
        image_tensor, image_np = self.preprocess_image(image_path)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        class_name = self.class_names[predicted_class]
        
        result = {
            'class': class_name,
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.class_names[0]: probs[0, 0].item(),
                self.class_names[1]: probs[0, 1].item()
            }
        }
        
        # Generate Grad-CAM if requested
        if return_gradcam:
            try:
                if cv2 is None:
                    raise ImportError("OpenCV (cv2) is not available in this environment")

                # Generate Grad-CAM heatmap
                cam = self.gradcam.generate_cam(image_tensor, predicted_class, self.device)
                
                # Resize to match image size
                cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
                
                # Create heatmap
                heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Superimpose on image
                superimposed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
                
                result['gradcam_heatmap'] = cam_resized
                result['gradcam_overlay'] = superimposed
                
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM: {e}")
                result['gradcam_heatmap'] = None
                result['gradcam_overlay'] = None
        
        return result
    
    def predict_batch(
        self,
        image_paths: list,
        return_gradcam: bool = False
    ) -> list:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths (list): List of image paths
            return_gradcam (bool): Whether to generate Grad-CAM
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, return_gradcam=return_gradcam)
            results.append(result)
        
        return results


def load_and_predict(
    image_path: str,
    model_path: str,
    model_name: str = 'resnet18'
) -> Dict:
    """
    Convenience function to load model and make prediction.
    
    Args:
        image_path (str): Path to image
        model_path (str): Path to model checkpoint
        model_name (str): Model architecture name
        
    Returns:
        Dict: Prediction results
    """
    predictor = PneumoniaPredictor(model_path, model_name=model_name)
    result = predictor.predict(image_path, return_gradcam=True)
    
    return result


if __name__ == '__main__':
    # Example usage
    model_path = './checkpoints/best_model.pt'
    image_path = './data/chest_xray/test/NORMAL/NORMAL2-IM-0003-0001.jpeg'
    
    result = load_and_predict(image_path, model_path)
    
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
