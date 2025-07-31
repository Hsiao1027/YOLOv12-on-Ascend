from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov12n.pt')
    metrics = model.val(
        batch=4,
        imgsz=960,
        data='.dataset.yaml',   
        split='val'                                            
    )
  
