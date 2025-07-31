from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/workspace/pdf2/yolov12/runs/detect/train14/weights/best.pt')
    metrics = model.val(
        batch=4,
        imgsz=960,
        data='./mix_dataset.yaml',   
        split='val'                                            
    )
  