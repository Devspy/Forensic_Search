from ultralytics import YOLO
import cv2
from datetime import datetime
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os

model = YOLO("Resource\\yolov8n.pt")

cap = cv2.VideoCapture("VideoFile\\video2.mp4")
unique_ids = set()
tracked_objects = {}
os.makedirs("crops", exist_ok=True)

frame_count = 0
SKIP_FRAMES = 1  # process 1 frame, skip 2

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # results = model(frame)

    # for result in results:
    #     for box in result.boxes:
    #         conf = float(box.conf[0])

    #         if conf > 0.50:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])

    #             cls_id = int(box.cls[0])
    #             label = f"{model.names[cls_id]} {conf:.2f}"

    #             cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
    #             cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)

    # cv2.imshow("YOLOv8", frame)

    # frame_count += 1

    # if frame_count % (SKIP_FRAMES + 1) != 0:
    #     continue

    results = model.track(frame,persist=True,conf=0.50,classes=[0, 2, 3, 5, 7],tracker="botsort.yaml",verbose=False)

    for result in results:
        for box in result.boxes:

            conf = float(box.conf[0])

            if conf > 0.50:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                # Get tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])
                    unique_ids.add(track_id)
                    object_name = model.names[cls_id]
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if track_id not in tracked_objects:
                        crop_path = f"crops/{track_id}.jpg"
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                             cv2.imwrite(crop_path, crop)
                        tracked_objects[track_id] = {"class": object_name,"first_seen": now,"last_seen": now,
                            "max_conf": conf,"image": crop_path}
                    else:
                        tracked_objects[track_id]["last_seen"] = now
                        tracked_objects[track_id]["max_conf"] = max(
                        tracked_objects[track_id]["max_conf"],
                        conf)
                else:
                    track_id = -1

                label = f"ID:{track_id} {model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                #cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
                cv2.putText(frame,f"Objects Count: {len(unique_ids)}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)

    cv2.imshow("YOLOv8 Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

pdf = SimpleDocTemplate("tracking_report.pdf")

styles = getSampleStyleSheet()

elements = []

# Report Title
elements.append(
    Paragraph("Forensic Detection Report", styles["Title"])
)

elements.append(Spacer(1, 15))

# Summary
elements.append(
    Paragraph(
        f"<b>Total Unique Objects Detected:</b> {len(unique_ids)}",
        styles["Heading2"]
    )
)

elements.append(Spacer(1, 20))

# Table Header
table_data = [[
    "Sl No",
    "Snapshot",
    "Object Type",
    "First Seen",
    "Last Seen",
    "Confidence"
]]

serial_no = 1

for track_id, info in tracked_objects.items():

    image_cell = "N/A"

    try:
        if os.path.exists(info["image"]):
            image_cell = Image(
                info["image"],
                width=60,
                height=60
            )
    except:
        image_cell = "N/A"

    table_data.append([
        str(serial_no),
        image_cell,
        info["class"].title(),
        info["first_seen"],
        info["last_seen"],
        f"{info['max_conf']:.2f}"
    ])

    serial_no += 1

# Create Table
table = Table(
    table_data,
    colWidths=[
        40,   # Sl No
        70,   # Snapshot
        80,   # Object Type
        120,  # First Seen
        120,  # Last Seen
        60    # Confidence
    ],
    rowHeights=[30] + [70] * (len(table_data) - 1)
)

# Table Styling
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),

    ('GRID', (0, 0), (-1, -1), 1, colors.black),

    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

    ('ROWBACKGROUNDS',
     (0, 1),
     (-1, -1),
     [colors.whitesmoke, colors.lightgrey])
]))

elements.append(table)

elements.append(Spacer(1, 20))

# Object Statistics
elements.append(
    Paragraph("Object Statistics", styles["Heading1"])
)

object_stats = {}

for track_id, info in tracked_objects.items():
    obj = info["class"]

    if obj not in object_stats:
        object_stats[obj] = 0

    object_stats[obj] += 1

stats_data = [["Object Type", "Count"]]

for obj, count in sorted(object_stats.items()):
    stats_data.append([
        obj.title(),
        str(count)
    ])

stats_table = Table(
    stats_data,
    colWidths=[200, 100]
)

stats_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER')
]))

elements.append(stats_table)

elements.append(Spacer(1, 20))

# Grand Total
elements.append(
    Paragraph(
        f"<b>Grand Total Objects Detected : {len(unique_ids)}</b>",
        styles["Heading2"]
    )
)

# Report Generation Time
elements.append(
    Paragraph(
        f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    )
)

pdf.build(elements)

print("\n===================================")
print(" PDF Report Generated Successfully")
print(" File : tracking_report.pdf")
print(f" Total Objects : {len(unique_ids)}")
print("===================================")