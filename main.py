from ultralytics import YOLO
import cv2

model = YOLO("Resource\\yolov8n.pt")

cap = cv2.VideoCapture("VideoFile\\video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])

            if conf > 0.50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)

    cv2.imshow("YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()