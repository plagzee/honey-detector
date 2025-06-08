import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import time
import threading

MODEL_PATH = "honey_detector_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


model = load_model()
class_names = ['honey', 'not honey']

detection_result = "Looking for honey..."

lock = threading.Lock()

def detect_honey(frame):
    global detection_result

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    input_tensor = transforms(pil_image).unsqueeze(0).to(device)

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

def detection_loop(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_honey(frame)
        time.sleep(1.5)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    detection_thread = threading.Thread(target=detection_loop, args=(cap,), daemon=True)
    detection_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, detection_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Honey Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()