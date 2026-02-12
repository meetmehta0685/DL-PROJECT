"""
Grad-CAM implementation for model interpretability.
Generates visual explanations of model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    Produces class-discriminative localization maps.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize GradCAM.
        
        Args:
            model (torch.nn.Module): Neural network model
            target_layer (torch.nn.Module): Target layer for feature extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._get_activation_hook())
        self.target_layer.register_full_backward_hook(self._get_gradient_hook())
    
    def _get_activation_hook(self):
        """Create a hook to get activations."""
        def hook(module, input, output):
            self.activations = output.detach()
        return hook
    
    def _get_gradient_hook(self):
        """Create a hook to get gradients."""
        def hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        return hook
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor (1, 3, H, W)
            target_class (int, optional): Target class for explanation
            device (torch.device): Device to use
            
        Returns:
            np.ndarray: Grad-CAM heatmap (H, W)
        """
        input_tensor = input_tensor.to(device)
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            # If no target class specified, use the predicted class
            if target_class is None:
                target_class = torch.argmax(output, dim=1).item()
            
            # Backward pass for target class
            self.model.zero_grad()
            class_loss = output[0, target_class]
            class_loss.backward()
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights: average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()
    
    def generate_cam_with_input(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ) -> np.ndarray:
        """
        Generate superimposed Grad-CAM on input image.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor (1, 3, H, W)
            target_class (int, optional): Target class
            device (torch.device): Device to use
            
        Returns:
            np.ndarray: Grad-CAM heatmap overlaid on input image
        """
        # Denormalize input image (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        
        input_img = input_tensor.clone()
        input_img = (input_img * std + mean).clamp(0, 1)
        
        # Convert to numpy for visualization
        img_np = input_img[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class, device)
        
        # Resize CAM to match input size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Superimpose on input image
        superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        return superimposed


class CNNMLPWrapper(nn.Module):
    """
    Wrap a CNN feature extractor and an MLP head into a single model.
    This lets Grad-CAM use CNN activations while the prediction comes from the MLP.
    """

    def __init__(self, cnn_backbone: nn.Module, mlp_head: nn.Module):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.mlp_head = mlp_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self.mlp_head(features)


def _infer_target_layer(cnn_backbone: nn.Module, model_name: Optional[str]) -> nn.Module:
    if hasattr(cnn_backbone, 'get_last_conv_layer'):
        return cnn_backbone.get_last_conv_layer()
    if model_name and model_name.lower() == 'resnet18' and hasattr(cnn_backbone, 'layer4'):
        return cnn_backbone.layer4[-1]
    if model_name and model_name.lower() == 'efficientnet_b0' and hasattr(cnn_backbone, 'features'):
        return cnn_backbone.features[-1]
    if hasattr(cnn_backbone, 'layer4'):
        return cnn_backbone.layer4[-1]
    if hasattr(cnn_backbone, 'features'):
        return cnn_backbone.features[-1]
    raise ValueError('Could not infer target layer for Grad-CAM. Please pass a model with a known conv layer.')


def create_gradcam_for_cnn_mlp(
    cnn_backbone: nn.Module,
    mlp_head: nn.Module,
    model_name: Optional[str] = 'resnet18'
) -> GradCAM:
    """
    Create a Grad-CAM explainer using a CNN feature extractor and an MLP head.

    Args:
        cnn_backbone (nn.Module): CNN feature extractor (e.g., ResNet18 with fc=Identity)
        mlp_head (nn.Module): MLP classifier head
        model_name (str, optional): Model name for target layer selection

    Returns:
        GradCAM: Grad-CAM explainer instance
    """
    wrapped_model = CNNMLPWrapper(cnn_backbone, mlp_head)
    target_layer = _infer_target_layer(cnn_backbone, model_name)
    return GradCAM(wrapped_model, target_layer)


def generate_cam_with_mlp_confidence(
    cnn_backbone: nn.Module,
    mlp_head: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
    model_name: Optional[str] = 'resnet18'
) -> Tuple[np.ndarray, float, int]:
    """
    Generate Grad-CAM heatmap using CNN features and return MLP confidence.

    Returns:
        Tuple[np.ndarray, float, int]: (heatmap, confidence, class_id)
    """
    wrapped_model = CNNMLPWrapper(cnn_backbone, mlp_head).to(device)
    wrapped_model.eval()

    with torch.no_grad():
        logits = wrapped_model(input_tensor.to(device))
        probs = F.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, target_class].item())

    gradcam = create_gradcam_for_cnn_mlp(cnn_backbone, mlp_head, model_name=model_name)
    cam = gradcam.generate_cam(input_tensor, target_class=target_class, device=device)
    return cam, confidence, target_class


def visualize_gradcam(
    image: np.ndarray,
    gradcam_heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    save_path: Optional[str] = None
):
    """
    Visualize Grad-CAM heatmap with image and prediction.
    
    Args:
        image (np.ndarray): Original image
        gradcam_heatmap (np.ndarray): Grad-CAM heatmap
        prediction (str): Prediction label
        confidence (float): Confidence score
        save_path (str, optional): Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(gradcam_heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Superimposed
    axes[2].imshow(gradcam_heatmap)
    axes[2].set_title(f'Prediction: {prediction}\nConfidence: {confidence:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.show()


def create_gradcam_explainer(
    model: torch.nn.Module,
    model_name: str = 'resnet18'
) -> GradCAM:
    """
    Create a Grad-CAM explainer for the model.
    
    Args:
        model (torch.nn.Module): Trained model
        model_name (str): Name of the model architecture
        
    Returns:
        GradCAM: Grad-CAM explainer instance
    """
    if model_name.lower() == 'resnet18':
        target_layer = model.get_last_conv_layer()
    elif model_name.lower() == 'efficientnet_b0':
        target_layer = model.get_last_conv_layer()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return GradCAM(model, target_layer)


if __name__ == '__main__':
    # Test GradCAM
    import torch
    from src.model import get_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model('resnet18')
    model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Create GradCAM
    gradcam = create_gradcam_explainer(model, 'resnet18')
    
    # Generate CAM
    cam = gradcam.generate_cam(x, device=device)
    
    print(f"Grad-CAM shape: {cam.shape}")
    print(f"Grad-CAM min: {cam.min():.4f}, max: {cam.max():.4f}")
