from ultralytics import YOLO
import cv2

model = YOLO(r"F:\projects\Python\YOLO\V8\yolov8\runs\detect\yolov8n_custom2\weights\best.pt")

img = cv2.imread(r"F:\projects\Python\YOLO\V8\yolov8\datasets\train\images\China_MotorBike_000171.jpg")

results = model(img)

# Получить аннотированное изображение
annotated_img = results[0].plot()

# Сохранить изображение с предсказаниями
output_path = 'output_image.jpg'
cv2.imwrite(output_path, annotated_img)

print(f'Предсказание сохранено в {output_path}')