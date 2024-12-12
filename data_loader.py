from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from embedding_extractor import EmbeddingExtractor

class DataLoader:
    def __init__(self, data_path="data/cifar10/"):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.embedding_extractor = EmbeddingExtractor()  # Initialize the extractor

    def download_cifar10(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
        ])
        dataset = CIFAR10(root=self.data_path, train=True, download=True, transform=transform)
        for idx, (image_tensor, label) in enumerate(dataset):
            # Convert tensor to PIL image
            image = transforms.ToPILImage()(image_tensor)
            image_path = os.path.join(self.data_path, f"image_{idx}.jpg")
            image.save(image_path)
            if idx >= 100:  # Limit to 100 images for quick testing
                break
        print("CIFAR-10 images downloaded and saved.")

    def load_images(self, folder_path):
        """Load images and extract embeddings."""
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png")):
                images.append(os.path.join(folder_path, filename))
        return images  # Return only the paths as a flat list

# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np
# from embedding_extractor import EmbeddingExtractor

# class DataLoader:
#     def __init__(self, data_path="data/cifar10/"):
#         self.data_path = data_path
#         os.makedirs(self.data_path, exist_ok=True)
#         self.embedding_extractor = EmbeddingExtractor()  # Initialize the extractor

#     def download_cifar10(self):
#         transform = transforms.Compose([
#             transforms.ToTensor(),  # Convert images to tensors
#         ])
#         dataset = CIFAR10(root=self.data_path, train=True, download=True, transform=transform)
#         for idx, (image_tensor, label) in enumerate(dataset):
#             # Convert tensor to PIL image
#             image = transforms.ToPILImage()(image_tensor)
#             image_path = os.path.join(self.data_path, f"image_{idx}.jpg")
#             image.save(image_path)
#             if idx >= 100:  # Limit to 100 images for quick testing
#                 break
#         print("CIFAR-10 images downloaded and saved.")

#     def load_images(self, folder_path):
#         images = []
#         embeddings = []
#         for filename in os.listdir(folder_path):
#             if filename.endswith((".jpg", ".png")):
#                 image_path = os.path.join(folder_path, filename)
#                 images.append(image_path)
#                 embedding = self.embedding_extractor.extract(image_path)
#                 embeddings.append(embedding)
#         return images, np.array(embeddings)
