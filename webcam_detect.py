import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import time
import threading

MODEL_PATH = "honey_detector_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms (same as during training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()
class_names = ['honey', 'not honey']

detection_result = "Looking for honey..."
lock = threading.Lock()

# Honey detection logic
def detect_honey(frame):
    global detection_result
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = image_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)

    label = class_names[pred.item()]
    confidence = conf.item()

    with lock:
        if label == 'honey' and confidence > 0.7:
            detection_result = "✅ Honey detected!"
        else:
            detection_result = "❌ No honey detected."

# Loop that runs detection every 1.5s in background
def detection_loop(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_honey(frame)
        time.sleep(1.5)

# Prompt to choose webcam
def select_camera():
    print("🔍 Scanning for webcams... Press 'q' to skip a camera.")

    for i in range(5):  # Try cameras 0 through 4
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        print(f"📸 Press 's' to select camera {i} or 'q' to skip.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(f"Camera {i}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                return i
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    print("❌ No camera selected. Exiting.")
    exit()

# Main logic
def main():
    cam_index = select_camera()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Error: Could not open selected camera")
        return

    detection_thread = threading.Thread(target=detection_loop, args=(cap,), daemon=True)
    detection_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            label = detection_result

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "✅" in label else (0, 0, 255), 2)
        cv2.imshow("Honey Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
