# YOLOv12-on-Ascend

## 📌 Overview

This repository provides a **PyTorch-based** implementation and training setup for YOLO series models on **Huawei Ascend NPUs** using the **torch_npu** extension. By leveraging the compute power of Ascend AI Processors, developers can train object detection models such as YOLOv12 efficiently within the Ascend ecosystem.

Ascend is a full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. For more information, please see [Ascend Community](https://www.hiascend.com/en/) or [Ascend Extension for Pytorch](https://github.com/Ascend/pytorch).

## 🔧 Modified Files

To support training on Ascend NPUs, the following files from the original Ultralytics YOLOv12 implementation have been **modified** or **extended**:

| File | Description | Status |
|------|-------------|--------|
| `train.py` | Modified to support `torch_npu`, including device selection, AMP, and HCCL | ✅ Extended |
| `train_ddp.py` | Modified to support distributed training on Ascend NPUs using HCCL and `torch.distributed` | ✅ Extended |
| `engine/` | Modified training logic in select files to support NPU device setup, AMP, and synchronization | ✅ Partly Replaced |
| `nn/` | Updated specific modules to ensure compatibility with fused layers and torch_npu execution | ✅ Partly Replaced |
| `utils/` | Added NPU device detection, precision handling, and utility functions in relevant utility scripts | ✅ Partly Replaced |

## ⚙️ Installation

```bash
# (optional) conda create -n yolov12-ascend python=3.10 -y && conda activate yolov12-ascend
pip install -r requirements.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 🚀 Training (Quick Start)

### Single-NPU
```bash
python train.py 
```

### Multi-NPU (DDP with HCCL)
```bash
torchrun --nproc_per_node=8 train_ddp.py
```


##  Acknowledgments
- Ultralytics YOLO
- Huawei Ascend / torch_npu teams

---

## License
Follow the upstream Ultralytics license and add-on notices for modified files.


> 🔄 All modified files are included in this repository. If you're using the official [Ultralytics YOLOv12](https://github.com/sunsmarterjie/yolov12) repo, you can **replace the corresponding files** with those provided here to enable Ascend NPU support.
