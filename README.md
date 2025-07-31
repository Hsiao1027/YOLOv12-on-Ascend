# YOLOv12-on-Ascend

## Overview

This repository provides a **PyTorch-based** implementation and training setup for YOLO series models on **Huawei Ascend NPUs** using the **torch_npu** extension. By leveraging the compute power of Ascend AI Processors, developers can train object detection models such as YOLOv12 efficiently within the Ascend ecosystem.

Ascend is a full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. For more information about Ascend, see [Ascend Community](https://www.hiascend.com/en/) and [Ascend Extension for Pytorch](https://github.com/Ascend/pytorch).

## 🔧 Modified Files

To support training on Ascend NPUs, the following files from the original Ultralytics YOLOv12 implementation have been **modified** or **extended**:

| File | Description | Status |
|------|-------------|--------|
| `train.py` | Modified to support `torch_npu`, including device selection, AMP, and HCCL | ✅ Replaced |
| `val.py` | Adjusted for NPU inference and evaluation on Ascend | ✅ Replaced |
| `ultralytics/engine/trainer.py` | Inserted NPU-specific logic and synchronization handling | ✅ Replaced |
| `ultralytics/nn/modules/block.py` | Resolved compatibility with custom fused layers on Ascend | ⚠️ Slight Modification |
| `ultralytics/yolo/utils/torch_utils.py` | Added support for `npu` device detection and conversion utilities | ✅ Replaced |
| `ultralytics/yolo/utils/__init__.py` | Registered device-specific hooks for NPU | ⚠️ Added few lines |

> 📌 All modified files are included in this repository. If you're using the official [Ultralytics YOLOv12](https://github.com/ultralytics/yolov12) repo, you can **replace the corresponding files** with those provided here to enable Ascend NPU support.

---

## 🔄 Integration Guide

To integrate the Ascend-compatible components into your existing Ultralytics environment:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourname/YOLOv12-on-Ascend.git
