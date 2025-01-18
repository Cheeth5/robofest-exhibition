from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model.train(
    data='/home/cheeth/Downloads/yolov8_version/final/data.yaml',#change this directory to the directory of the data.yaml
    epochs=100, 
    imgsz=640,
    project='/home/cheeth/Downloads/yolov8_version/output',  
    name='test_training',
    save=True,  # Force saving
    save_period=1  # Save weights every epoch
)
