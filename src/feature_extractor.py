import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Base abstract class for all feature extractors"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = self._get_transform()
        self.model = self._load_model().to(self.device)
        self.model.eval()
        
    @abstractmethod
    def _load_model(self):
        """Load the model architecture"""
        pass
    
    def _get_transform(self):
        """Default image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_from_url(self, image_url):
        """Download image from URL and extract features"""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to download image: {response.status_code}")
                return None
                
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return self.extract_from_image(image)
        except Exception as e:
            print(f"Error processing image URL {image_url}: {e}")
            return None
    
    def extract_from_image(self, image):
        """Extract features from a PIL Image"""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                
            # Normalize the features
            features = features.squeeze().cpu().numpy()
            normalized_features = features / np.linalg.norm(features)
            return normalized_features.astype(np.float32)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None


class ResNet50Extractor(BaseFeatureExtractor):
    """Feature extractor using ResNet50"""
    
    def _load_model(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the classification layer
        return torch.nn.Sequential(*list(model.children())[:-1])


class EfficientNetExtractor(BaseFeatureExtractor):
    """Feature extractor using EfficientNet-B0"""
    
    def _load_model(self):
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove the classification layer
        return torch.nn.Sequential(*list(model.children())[:-1])


class MobileNetExtractor(BaseFeatureExtractor):
    """Feature extractor using MobileNetV3 Small"""
    
    def _load_model(self):
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove the classification layer and global average pooling
        return torch.nn.Sequential(*list(model.children())[:-2])
    
    def extract_from_image(self, image):
        """Override to handle the different output shape of MobileNet"""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                # Apply global average pooling
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                
            # Normalize the features
            features = features.view(features.size(0), -1).squeeze().cpu().numpy()
            normalized_features = features / np.linalg.norm(features)
            return normalized_features.astype(np.float32)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None


def get_feature_extractor(model_name="efficientnet"):
    """Factory function to get the appropriate feature extractor"""
    extractors = {
        "resnet50": ResNet50Extractor,
        "efficientnet": EfficientNetExtractor,
        "mobilenet": MobileNetExtractor
    }
    
    if model_name not in extractors:
        print(f"Model {model_name} not supported. Using EfficientNet instead.")
        model_name = "efficientnet"
        
    return extractors[model_name]()