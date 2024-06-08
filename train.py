from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='custom_data.yaml',
   imgsz=640,
   epochs=1,
   batch=8,
   device='gpu',
   name='yolov8n_custom'),
   
# Запустить тренировку
