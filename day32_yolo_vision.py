import cv2
import urllib.request
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import torch

print("--- Day 32: Object Detection with YOLO & Apple Silicon ---\n")

project_dir = Path(__file__).parent
image_path = project_dir / "sample_vision.jpg"
output_path = project_dir / "YOLO_V8.png"

# 1. Hardware Verification
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Routing YOLO Vision to hardware: {device.upper()}\n")

# 2. Load the YOLO Model
# We use YOLOv8n (the 'n' stands for nano - it is incredibly fast and lightweight)
print("📥 Loading YOLOv8-Nano model...")
model = YOLO("yolov8n.pt")

# 3. Fetch a Sample Image
# Let's download a sample image of people in a room (to simulate a hospital waiting room)
image_url = "https://ultralytics.com/images/zidane.jpg"
print(f"📸 Downloading sample image to {image_path.name}...")
urllib.request.urlretrieve(image_url, image_path)

# 4. Run Inference (The Forward Pass)
# We pass the image to the model and explicitly tell it to use the M5 chip
print("\n⚙️ Running Object Detection...")
results = model.predict(
    source=str(image_path),
    device=device,
    conf=0.5,
    verbose=False
)

# 5. Extracting the Data
# YOLO doesn't just return an image; it returns the raw math (coordinates and probabilities)
result = results[0]
boxes = result.boxes

print(f"\n✅ Detection Complete! Found {len(boxes)} objects.")

for i, box in enumerate(boxes):
    # Extract the coordinates of the bounding box
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    # Extract the confidence score
    confidence = box.conf[0].item()
    # Extract the class ID (e.g., 0 = person, 67 = cell phone)
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]

    print(f"   - Object {i+1}: {class_name.upper()} (Confidence: {confidence*100:.1f}%) at coordinates [{int(x1)}, {int(y1)}]")

# 6. Visualizing the Result
print("\n🎨 Saving Bounding Box Visualization...")

# Ultralytics has a built-in method to draw the boxes on the image
annotated_image = result.plot()

# Convert colors from BGR (OpenCV standard) to RGB (Matplotlib standard)
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(annotated_image_rgb)
plt.axis("off")
plt.title(f"YOLOv8 Detection (Hardware: {device.upper()})")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved YOLO detection visualization to: {output_path.name}")
