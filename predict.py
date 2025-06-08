import torch
from torchvision import models, transforms
from PIL import Image
import sys

# Path to your saved model weights
MODEL_PATH = "honey_detector.pt"

# Device (GPU if available else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transforms as during training (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225]     # ImageNet stds
    )
])

# Load the model architecture and weights
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Predict function: returns class index and confidence
def predict_image(image_path, model):
    # Open image
    image = Image.open(image_path).convert("RGB")
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()

# Map class indices to names (must match your folder names in training)
class_names = ['honey', 'not_honey']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    pred_idx, confidence = predict_image(image_path, model)

    print(f"Prediction: {class_names[pred_idx]} (Confidence: {confidence*100:.2f}%)")
