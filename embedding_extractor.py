from torchvision import models, transforms
from PIL import Image
import torch

class EmbeddingExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove the classification head
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(tensor).squeeze()  # Remove batch and channel dimensions
        return embedding.numpy().flatten()
