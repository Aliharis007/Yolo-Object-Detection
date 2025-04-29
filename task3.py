import cv2 
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("Real-time Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
