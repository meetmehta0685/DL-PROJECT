"""
CNN model architecture using transfer learning.
Supports ResNet18 and EfficientNet-B0.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Pneumonia(nn.Module):
    """
    ResNet18 model with custom classifier for pneumonia detection.
    Uses transfer learning from ImageNet pretrained weights.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize ResNet18 model.

        Args:
            num_classes (int): Number of output classes (default: 2)
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(ResNet18Pneumonia, self).__init__()

        # Load pretrained ResNet18
        if pretrained:
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as exc:
                print(f"Warning: could not download ResNet18 weights ({exc}). Using random init.")
                self.model = models.resnet18(weights=None)
        else:
            self.model = models.resnet18(weights=None)
        
        # Get the number of input features for the classifier
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        return self.model(x)
    
    def get_last_conv_layer(self) -> torch.nn.Module:
        """
        Get the last convolutional layer for GradCAM.
        
        Returns:
            torch.nn.Module: Last conv layer
        """
        return self.model.layer4[-1]
    
    def freeze_backbone(self):
        """Freeze the backbone layers for transfer learning."""
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone layers."""
        for param in self.model.layer1.parameters():
            param.requires_grad = True
        for param in self.model.layer2.parameters():
            param.requires_grad = True


class EfficientNetPneumonia(nn.Module):
    """
    EfficientNet-B0 model with custom classifier for pneumonia detection.
    Uses transfer learning from ImageNet pretrained weights.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize EfficientNet-B0 model.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            pretrained (bool): Use ImageNet pretrained weights
        """
        super(EfficientNetPneumonia, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            try:
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            except Exception as exc:
                print(f"Warning: could not download EfficientNet-B0 weights ({exc}). Using random init.")
                self.model = models.efficientnet_b0(weights=None)
        else:
            self.model = models.efficientnet_b0(weights=None)
        
        # Get the number of input features for the classifier
        num_features = self.model.classifier[1].in_features
        
        # Replace the final classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        return self.model(x)
    
    def get_last_conv_layer(self) -> torch.nn.Module:
        """
        Get the last convolutional layer for GradCAM.
        
        Returns:
            torch.nn.Module: Last conv layer
        """
        return self.model.features[-1]
    
    def freeze_backbone(self):
        """Freeze the backbone layers for transfer learning."""
        for param in self.model.features[:-1].parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone layers."""
        for param in self.model.features.parameters():
            param.requires_grad = True


def get_model(model_name: str = 'resnet18', num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function to get a model.
    
    Args:
        model_name (str): Model name ('resnet18' or 'efficientnet_b0')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to initialize with ImageNet pretrained weights
        
    Returns:
        nn.Module: Model instance
    """
    if model_name.lower() == 'resnet18':
        return ResNet18Pneumonia(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnet_b0':
        return EfficientNetPneumonia(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ResNet18
    model_resnet = get_model('resnet18')
    model_resnet.to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model_resnet(x)
    print(f"ResNet18 output shape: {output.shape}")
    
    # Test EfficientNet
    model_efficientnet = get_model('efficientnet_b0')
    model_efficientnet.to(device)
    output = model_efficientnet(x)
    print(f"EfficientNet-B0 output shape: {output.shape}")
