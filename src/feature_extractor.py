import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np


class FeatureExtractor:
    def __init__(self):
        """Initialize ResNet50 for feature extraction."""
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final classification layer
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract feature vector from an image."""
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image).squeeze().numpy()
        return features / np.linalg.norm(features)  # Normalize to unit norm
