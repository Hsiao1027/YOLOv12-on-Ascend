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
        dist.init_process_group(backend="hccl", rank=local_rank)

        dist.barrier()
        time.sleep(3)

        if local_rank == 0:
            print("[Rank 0] 🚀 Starting model construction...")
        dist.barrier()

        model = YOLO("./ultralytics/cfg/models/v12/yolov12.yaml")
        model.model.to(device)

        dist.barrier()
        time.sleep(2)

        model.train(
            data="/root/workspace/pdf2/yolov12/mix_dataset.yaml",
            epochs=100,
            imgsz=1024,
            batch=10,        
            workers=4,        
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

    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    main()
