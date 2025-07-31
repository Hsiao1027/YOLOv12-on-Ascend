import os
import torch
import torch_npu
from ultralytics import YOLO

def main():

    os.environ["ASCEND_VISIBLE_DEVICES"] = "0"

    torch.npu.set_device(0)
    device = torch.device("npu")

    print(f"[Single Card] Using device: {device}")

    model = YOLO("./ultralytics/cfg/models/v12/yolov12.yaml")
    model.model.to(device)

    model.train(
        data="./datasets/data.yaml",
        epochs=2,
        imgsz=640,
        batch=8,
        device=device.type,  # 'npu'
        workers=2,
        verbose=True,
        amp=False,
        save=True
    )

if __name__ == "__main__":
    main()
