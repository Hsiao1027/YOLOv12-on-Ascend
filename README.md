# YOLOv12-on-Ascend

## 📌 Overview

This repository provides a **PyTorch-based** implementation and training setup for YOLO series models on **Huawei Ascend NPUs** using the **torch_npu** extension. By leveraging the compute power of Ascend AI Processors, developers can train object detection models such as YOLOv12 efficiently within the Ascend ecosystem.

Ascend is a full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. For more information about Ascend, see [Ascend Community](https://www.hiascend.com/en/) and [Ascend Extension for Pytorch](https://github.com/Ascend/pytorch).

## 🔧 Modified Files

To support training on Ascend NPUs, the following files from the original Ultralytics YOLOv12 implementation have been **modified** or **extended**:

| File | Description | Status |
|------|-------------|--------|
| `train.py` | Modified to support `torch_npu`, including device selection, AMP, and HCCL | ✅ Extended |
| `train_ddp.py` | Modified to support distributed training on Ascend NPUs using HCCL and `torch.distributed` | ✅ Extended |
| `engine/` | Modified training logic in select files to support NPU device setup, AMP, and synchronization | ✅ Partly Replaced |
| `nn/` | Updated specific modules to ensure compatibility with fused layers and torch_npu execution | ✅ Partly Replaced |
| `utils/` | Added NPU device detection, precision handling, and utility functions in relevant utility scripts | ✅ Partly Replaced |


> 🔄 All modified files are included in this repository. If you're using the official [Ultralytics YOLOv12](https://github.com/sunsmarterjie/yolov12) repo, you can **replace the corresponding files** with those provided here to enable Ascend NPU support.
