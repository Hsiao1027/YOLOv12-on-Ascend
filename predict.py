import os
import torch
import torch_npu
from ultralytics import YOLO
import cv2

# 設定裝置（NPU 優先）
device = torch.device("npu" if torch.npu.is_available() else "cpu")

# 載入模型並轉移到 NPU
model = YOLO("/root/workspace/pdf2/yolov12/runs/detect/Pretrained_DocLayNet/weights/best.pt")
model.model.to(device)

# 設定資料夾路徑
image_folder = "./datasets/test/images"
output_folder = "./results/doclaynet0722"
os.makedirs(output_folder, exist_ok=True)

# 支援的圖像副檔名
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# 讀取資料夾中的每一張圖片
for filename in os.listdir(image_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)

        # 推論（不需要指定 device）
        results = model.predict(
            source=img,
            imgsz=960,
            conf=0.35,
            verbose=True,
        )

        # 繪製預測框並儲存
        annotated_frame = results[0].plot()
        output_path = os.path.join(output_folder, f"pred_{filename}")
        cv2.imwrite(output_path, annotated_frame)
        print(f"✅ Saved: {output_path}")
