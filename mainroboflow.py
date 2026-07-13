from ultralytics import RTDETR
import cv2

# Load RT-DETR model
# First run downloads the weights; afterwards you can use them offline.
model = RTDETR("Resource\\rtdetr-l.pt")
cap = cv2.VideoCapture("VideoFile\\video2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # COCO classes:# 0 = person # 2 = car
            if cls not in [0, 2]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("RT-DETR Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()