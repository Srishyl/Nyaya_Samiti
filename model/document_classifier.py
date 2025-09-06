import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=2): # 0 for Aadhaar, 1 for Passport
        super(DocumentClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Assuming input image size is 224x224, after two max pools (downsampling by 4),
        # the feature map size would be 64 * 56 * 56 (adjust based on actual input size)
        # For simplicity, let's assume a generic flattened size for the FC layer.
        # This will need to be adjusted once actual input size and feature map size are known.
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128), # Placeholder, will need dynamic calculation or specific input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the features
        x = self.classifier(x)
        return x

    def preprocess_image(self, pil_image):
        # Define the transformations for the input image
        # Resize to a common size, convert to tensor, normalize
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Common input size for many CNNs
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
        ])
        return transform(pil_image).unsqueeze(0) # Add batch dimension

    def predict(self, pil_image):
        # In a real scenario, the model would be loaded with trained weights.
        # For now, this is a placeholder for the prediction logic.
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            input_tensor = self.preprocess_image(pil_image)
            outputs = self(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), probabilities.squeeze().tolist()

# Example Usage (for testing purposes, not part of the integrated app logic directly)
if __name__ == "__main__":
    # Create a dummy image
    dummy_image = Image.new('RGB', (400, 300), color = 'red')

    classifier = DocumentClassifier(num_classes=2)
    # In a real scenario, you'd load state_dict here:
    # classifier.load_state_dict(torch.load('path_to_trained_model.pth'))

    predicted_class, probabilities = classifier.predict(dummy_image)
    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

    # Map class index to document type (this would be based on training)
    class_names = {0: "Aadhaar", 1: "Passport"}
    print(f"Predicted Document Type: {class_names.get(predicted_class, 'Unknown')}")
