
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# Error Level Analysis (ELA) implementation
def el-image(image_path, quality=90):
    original_image = Image.open(image_path).convert("RGB")
    temp_filename = "temp_el-image.jpg"
    original_image.save(temp_filename, 'JPEG', quality=quality)
    
    recompressed_image = Image.open(temp_filename)
    ela_result = ImageChops.difference(original_image, recompressed_image)
    
    extrema = ela_result.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: # Avoid division by zero
        max_diff = 1
    scale = 255.0 / max_diff
    ela_result = ImageEnhance.Brightness(ela_result).enhance(scale)
    
    os.remove(temp_filename)
    return ela_result

# Simple CNN for anomaly detection based on ELA
class ELA_CNN(nn.Module):
    def __init__(self, num_classes=2): # For binary classification: genuine/tampered
        super(ELA_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        # Placeholder for linear layer input size, will depend on image size
        # For 224x224 input, after 3 MaxPool2d(2,2), spatial dims become 224/8 x 224/8 = 28x28
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Example Usage and Setup Notes
    # 1. Dataset: Tampered and genuine images for training.
    # 2. Preprocessing: Apply ELA to all images before feeding to CNN.
    # 3. Training: Standard CNN training with CrossEntropyLoss.

    # Create a dummy image for ELA
    dummy_image_path = "dummy_document_for_ela.jpg"
    Image.new('RGB', (224, 224), color = 'red').save(dummy_image_path)

    # Generate ELA image
    print(f"Generating ELA for {dummy_image_path}...")
    ela_output_img = el-image(dummy_image_path)
    ela_output_img.save("dummy_ela_output.jpg")
    print("ELA image saved to dummy_ela_output.jpg")

    # Prepare ELA image for CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert PIL Image (ELA output) to tensor
    ela_tensor = transform(ela_output_img).unsqueeze(0)

    # Initialize ELA_CNN
    model = ELA_CNN(num_classes=2)

    # Forward pass
    output = model(ela_tensor)
    print("ELA_CNN output shape:", output.shape)

    # Clean up dummy image
    os.remove(dummy_image_path)
    os.remove("dummy_ela_output.jpg")
    print("Noise Inconsistency Model (ELA + CNN) outlined successfully.")
