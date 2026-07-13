import cv2
import os
import shutil
import numpy as np
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

    def resize_crop(self, crop, min_size=256):
        """Resize crop to minimum size with sharpening for clarity"""
        height, width = crop.shape[:2]
        
        # If already larger than minimum, still apply sharpening
        if height >= min_size and width >= min_size:
            return self.sharpen_image(crop)
        
        # Calculate scale factor based on smaller dimension
        scale = min_size / min(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize using LANCZOS4 - best for upscaling
        resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply sharpening to reduce blur
        sharpened = self.sharpen_image(resized)
        return sharpened

    def sharpen_image(self, image):
        """Apply unsharp masking to sharpen the image"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_float, (0, 0), 2.0)
        
        # Calculate sharpened image using unsharp mask
        sharpened = cv2.addWeighted(img_float, 1.5, blurred, -0.5, 0)
        
        # Clip values to valid range and convert back to uint8
        sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
        return sharpened

    def save_best_images(self):
        """Save the best quality image for each tracked object"""
        for object_id, obj_data in self.tracked_objects.items():
            if obj_data["crop_data"] is not None:
                # Create snapshot path with best frame number
                snapshot_path = os.path.join(
                    self.snapshot_folder,
                    f"{object_id}_best_frame_{obj_data['best_frame']}.jpg"
                )
                
                # Save with high quality (95%)
                cv2.imwrite(snapshot_path, obj_data["crop_data"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Update the image path in tracked objects
                obj_data["image"] = snapshot_path
                
                # Clean up memory
                obj_data["crop_data"] = None

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
                            # Create new object entry
                            crop = frame[y1:y2, x1:x2]
                            crop = self.resize_crop(crop, min_size=256)
                            
                            self.tracked_objects[object_id] = {
                                "class": label,
                                "max_conf": confidence,
                                "image": None,  # Will be set when we save
                                "crop_data": crop,  # Store crop in memory
                                "first_seen": current_time,
                                "last_seen": current_time,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "best_frame": self.frame_count
                            }
                        else:
                            # Update existing object
                            self.tracked_objects[object_id]["last_seen"] = current_time
                            self.tracked_objects[object_id]["x1"] = x1
                            self.tracked_objects[object_id]["y1"] = y1
                            self.tracked_objects[object_id]["x2"] = x2
                            self.tracked_objects[object_id]["y2"] = y2
                            
                            # Update if we found a better quality detection
                            if confidence > self.tracked_objects[object_id]["max_conf"]:
                                self.tracked_objects[object_id]["max_conf"] = confidence
                                self.tracked_objects[object_id]["best_frame"] = self.frame_count
                                
                                # Delete previous image file if it exists
                                if self.tracked_objects[object_id]["image"] and os.path.exists(self.tracked_objects[object_id]["image"]):
                                    try:
                                        os.remove(self.tracked_objects[object_id]["image"])
                                    except:
                                        pass
                                
                                # Update with new better quality crop
                                crop = frame[y1:y2, x1:x2]
                                crop = self.resize_crop(crop, min_size=256)
                                self.tracked_objects[object_id]["crop_data"] = crop

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
        
        # Save best quality image for each tracked object
        self.save_best_images()
        
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