import os
import time
import torch
import torch_npu
import torch.distributed as dist
from ultralytics import YOLO
import traceback

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    print(f"[Rank {local_rank}] ✅ Using device: {device}")

    try:
        # ✅ 初始化 HCCL 分散式通訊
        dist.init_process_group(backend="hccl", rank=local_rank)

        # ✅ 等待所有卡都完成通訊建立
        dist.barrier()
        time.sleep(3)

        # ✅ 加入強同步防止 DataLoader 卡住
        if local_rank == 0:
            print("[Rank 0] 🚀 Starting model construction...")
        dist.barrier()

        # ✅ 建立模型，不使用預訓練權重
        # model = YOLO("./ultralytics/cfg/models/v12/yolov12.yaml")
        model = YOLO("./runs/detect/DocLayNet_epoch50/weights/best.pt")
        model.model.to(device)

        # ✅ 再次同步，確保所有 Rank 完成初始化
        dist.barrier()
        time.sleep(2)

        # ✅ 訓練設定
        model.train(
            data="/root/workspace/pdf2/yolov12/mix_dataset.yaml",
            epochs=100,
            imgsz=1024,
            batch=10,         # 每卡 batch
            workers=4,        # 請依據 CPU 增加
            device=device.type,
            resume=False,
            amp=False,
            save=True,
            val=False,
            verbose=(local_rank == 0)
        )

    except Exception as e:
        print(f"[Rank {local_rank}] ❌ Exception occurred:")
        traceback.print_exc()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {local_rank}] 🔚 Destroyed process group.")

if __name__ == "__main__":
    # ✅ 強化初始化設定
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

    # 🔧 避免 dataloader 被 preload 影響（視情況決定是否加入）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    main()
