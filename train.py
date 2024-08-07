from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8m.pt')
 
# Training.
results = model.train(
   data='custom_data.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   device='gpu',
   name='yolov8m_custom'),
   
# Запустить тренировку
