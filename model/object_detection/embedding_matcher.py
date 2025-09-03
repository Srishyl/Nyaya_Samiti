
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class EmbeddingMatcher:
    def __init__(self, model_name='resnet18', device='cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model = self._load_embedding_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_embedding_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1]) # Remove the final classification layer
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Identity() # Remove the final classification layer
            model.avgpool = nn.Identity() # Remove avgpool to get features before pooling
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # For EfficientNet, we need to flatten after the features
        if model_name == 'efficientnet_b0':
            return nn.Sequential(model, nn.Flatten())
        return model

    def get_embedding(self, pil_image):
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu().numpy().flatten()

    def match_embeddings(self, embedding1, embedding2, threshold=0.8):
        similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        return similarity, similarity > threshold

if __name__ == '__main__':
    # Example Usage and Setup Notes
    # 1. Install necessary libraries: pip install torch torchvision scikit-learn pillow
    # 2. Prepare your known stamp/seal templates and new detected stamp/seal regions.

    matcher = EmbeddingMatcher(model_name='resnet18') # or 'efficientnet_b0'

    # Create dummy images for demonstration
    dummy_template = Image.new('RGB', (224, 224), color = 'blue')
    dummy_detected = Image.new('RGB', (224, 224), color = 'blue')
    dummy_different = Image.new('RGB', (224, 224), color = 'green')

    print(f"Getting embedding for template image...")
    template_embedding = matcher.get_embedding(dummy_template)

    print(f"Getting embedding for detected image...")
    detected_embedding = matcher.get_embedding(dummy_detected)

    print(f"Getting embedding for different image...")
    different_embedding = matcher.get_embedding(dummy_different)

    # Match similar images
    similarity_same, is_match_same = matcher.match_embeddings(template_embedding, detected_embedding)
    print(f"Similarity between template and detected (same): {similarity_same:.4f}, Match: {is_match_same}")

    # Match different images
    similarity_diff, is_match_diff = matcher.match_embeddings(template_embedding, different_embedding)
    print(f"Similarity between template and different: {similarity_diff:.4f}, Match: {is_match_diff}")

    # Clean up dummy images
    # os.remove(dummy_template_path)
    # os.remove(dummy_detected_path)
    # os.remove(dummy_different_path)

    print("Embedding matcher outlined successfully.")
