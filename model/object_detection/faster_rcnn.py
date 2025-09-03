
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np

# Placeholder for model and utility functions
class StampSealDetector:
    def __init__(self, num_classes=2, device='cuda'):  # num_classes: 1 for stamp/seal + 1 for background
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model = self._load_model(num_classes)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, num_classes):
        # Load pre-trained Faster R-CNN model with a ResNet50-FPN backbone
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace the classifier with a new one for fine-tuning
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        return model

    def detect(self, pil_image, threshold=0.7):
        image_tensor = F.to_tensor(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image_tensor)

        detections = []
        if prediction and len(prediction) > 0:
            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']

            for i in range(len(scores)):
                if scores[i] > threshold:
                    box = boxes[i].cpu().numpy().astype(int)
                    label = labels[i].item()
                    score = scores[i].item()
                    detections.append({
                        'box': box.tolist(),
                        'label': label,
                        'score': score
                    })
        return detections

    def visualize_detections(self, original_pil_image, detections, output_path="output.jpg"):
        img_np = np.array(original_pil_image)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for det in detections:
            box = det['box']
            score = det['score']
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Red rectangle
            cv2.putText(img, f"Seal: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)
        print(f"Detection results saved to {output_path}")

if __name__ == '__main__':
    # Example Usage (replace with actual image path)
    # For this example to run, you will need a sample image with a stamp/seal
    # and a pre-trained model checkpoint if you are not training from scratch.
    # For demonstration, we will assume a generic image and a fresh model.
    
    # Setup steps:
    # 1. Install necessary libraries: pip install torch torchvision opencv-python pillow
    # 2. Prepare your dataset for stamp/seal detection (images and annotations).
    # 3. Train the model using your dataset (training logic not included in this file).
    # 4. Save the trained model's state_dict.

    # Example: Create a dummy image for testing
    dummy_image_path = "dummy_document.jpg"
    dummy_image = Image.new('RGB', (800, 600), color = (255, 255, 255))
    # Add a simple red rectangle as a 'stamp' for testing purposes
    for x in range(100, 200):
        for y in range(100, 150):
            dummy_image.putpixel((x, y), (255, 0, 0))
    dummy_image.save(dummy_image_path)

    detector = StampSealDetector(num_classes=2) # 1 class (stamp/seal) + background
    # If you have a trained model, load its state dict:
    # detector.model.load_state_dict(torch.load("path/to/your/trained_model.pth"))

    print(f"Detecting stamps/seals...")
    detections = detector.detect(dummy_image)
    if detections:
        print("Detections found:")
        for det in detections:
            print(f"  Box: {det['box']}, Label: {det['label']}, Score: {det['score']:.2f}")
        detector.visualize_detections(dummy_image, detections, "detected_stamps_seals.jpg")
    else:
        print("No stamps or seals detected.")
