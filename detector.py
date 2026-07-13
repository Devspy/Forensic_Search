import cv2
import os
import shutil
from pathlib import Path
from datetime import datetime
from report_generator import ReportGenerator

class RTDETRVideoProcessor:

    def __init__(self, model, video_path, manager):

        self.model = model
        self.video_path = video_path
        self.manager = manager
        self.unique_ids = set()
        self.tracked_objects = {}
        self.snapshot_folder = os.path.join("Snapshots",Path(self.video_path).stem)
        os.makedirs(self.snapshot_folder, exist_ok=True)
        self.window_name = video_path
        self.frame_count = 0
        self.fps = 0

    def get_video_time(self, milliseconds):
        """Convert milliseconds to video timestamp in HH:MM:SS format"""
        total_seconds = milliseconds / 1000.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def process_video(self):

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            self.manager.remove_video(self.video_path)
            return

        # Get FPS from video
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            self.frame_count += 1

            # Get current video time - try using video capture position, fallback to frame count
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec > 0:
                current_time = self.get_video_time(pos_msec)
            else:
                # Fallback: calculate from frame count and FPS
                if self.fps > 0:
                    seconds = (self.frame_count - 1) / self.fps
                    milliseconds = seconds * 1000
                    current_time = self.get_video_time(milliseconds)
                else:
                    current_time = "00:00:00"

            # Use botsort.yaml tracker for persistent tracking
            results = self.model.track(
                frame,
                persist=True,
                conf=0.4,
                classes=[0, 2],
                tracker="botsort.yaml",
                verbose=False
            )

            for result in results:

                for box in result.boxes:

                    cls = int(box.cls[0])

                    if cls not in [0, 2]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = self.model.names[cls]
                    confidence = float(box.conf[0])

                    # Get tracking ID from botsort
                    if box.id is not None:
                        track_id = int(box.id[0])
                        object_id = f"OBJ_{track_id:04d}"
                        self.unique_ids.add(object_id)

                        # Create or update tracked object
                        if object_id not in self.tracked_objects:
                            # Save snapshot for new object
                            crop = frame[y1:y2, x1:x2]
                            snapshot_path = os.path.join(
                                self.snapshot_folder,
                                f"{object_id}_frame_{self.frame_count}.jpg"
                            )
                            cv2.imwrite(snapshot_path, crop)

                            # Store new object information
                            self.tracked_objects[object_id] = {
                                "class": label,
                                "max_conf": confidence,
                                "image": snapshot_path,
                                "first_seen": current_time,
                                "last_seen": current_time,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            }
                        else:
                            # Update existing object
                            self.tracked_objects[object_id]["last_seen"] = current_time
                            self.tracked_objects[object_id]["x1"] = x1
                            self.tracked_objects[object_id]["y1"] = y1
                            self.tracked_objects[object_id]["x2"] = x2
                            self.tracked_objects[object_id]["y2"] = y2
                            # Update confidence if this detection is better
                            if confidence > self.tracked_objects[object_id]["max_conf"]:
                                self.tracked_objects[object_id]["max_conf"] = confidence

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame,
                                f"{object_id} - {label} ({confidence:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2)

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        ReportGenerator.generate(video_path=self.video_path,tracked_objects=self.tracked_objects,unique_ids=self.unique_ids)
        
        # Delete snapshot folder after report generation
        try:
            if os.path.exists(self.snapshot_folder):
                shutil.rmtree(self.snapshot_folder)
                print(f"Deleted snapshot folder: {self.snapshot_folder}")
        except Exception as e:
            print(f"Error deleting snapshot folder: {e}")
        
        # Remove this video from manager
        self.manager.remove_video(self.video_path)