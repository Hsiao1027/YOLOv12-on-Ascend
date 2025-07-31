import os
import torch
import torch_npu
from ultralytics import YOLO

def main():
    # 限制只能看到 NPU:0（從源頭避免多卡被初始化）
    os.environ["ASCEND_VISIBLE_DEVICES"] = "0"

    # 設定單卡 NPU
    torch.npu.set_device(0)
    device = torch.device("npu")

    print(f"[Single Card] Using device: {device}")

    # 載入模型架構（不載入權重）
    model = YOLO("./ultralytics/cfg/models/v12/yolov12.yaml")
    model.model.to(device)

    # 開始訓練
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
