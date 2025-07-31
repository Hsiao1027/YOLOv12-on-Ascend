import os
import torch
import torch_npu
from ultralytics import YOLO
import cv2

device = torch.device("npu" if torch.npu.is_available() else "cpu")

model = YOLO("yolov12n.pt")
model.model.to(device)

image_folder = "./datasets/test/images"
output_folder = "./results"
os.makedirs(output_folder, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
for filename in os.listdir(image_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)

        results = model.predict(
            source=img,
            imgsz=960,
            conf=0.35,
            verbose=True,
        )

        annotated_frame = results[0].plot()
        output_path = os.path.join(output_folder, f"pred_{filename}")
        cv2.imwrite(output_path, annotated_frame)
        print(f"✅ Saved: {output_path}")
