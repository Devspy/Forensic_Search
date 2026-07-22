import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from report_generator import ReportGenerator

class RTDETRVideoProcessor:

    def __init__(self, model, video_path, manager, search_object=None, search_color=None):

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
        self.search_object = (search_object or "").strip().lower()
        self.search_color = self._normalize_color_name(search_color)
        self.default_class_ids = [0, 2, 5, 7, 15, 16]
        self.class_name_map = {
            "person": 0,
            "car": 2,
            "cat": 15,
            "dog": 16,
            "bicycle": 1,
            "motorbike": 3,
            "bus": 5,
            "truck": 7,
        }
        self.allowed_class_ids = self.get_allowed_class_ids()

        if self.search_object or self.search_color:
            print(f"Search filter -> object: {self.search_object or 'all'}, color: {self.search_color or 'any'}")

    def _normalize_color_name(self, color_name):
        if not color_name:
            return ""
        return color_name.strip().lower().replace("grey", "gray")

    def get_allowed_class_ids(self):
        if self.search_object:
            mapped_class_id = self.class_name_map.get(self.search_object)
            if mapped_class_id is not None:
                return [mapped_class_id]
        return self.default_class_ids

    def get_dominant_color_name(self, crop):
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = np.mean(hsv[:, :, 0])
        saturation = np.mean(hsv[:, :, 1])
        value = np.mean(hsv[:, :, 2])

        if saturation < 40:
            if value < 60:
                return "black"
            if value > 180:
                return "white"
            return "gray"

        if hue < 20 or hue > 160:
            return "red"
        if hue < 40:
            return "yellow"
        if hue < 80:
            return "green"
        if hue < 140:
            return "blue"
        return "purple"

    def matches_user_request(self, label, crop):
        label_name = (label or "").strip().lower()

        if self.search_object and label_name != self.search_object:
            return False

        if self.search_color:
            dominant_color = self.get_dominant_color_name(crop)
            if dominant_color != self.search_color:
                return False

        return True

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

        start_time = datetime.now()
        print(f"Video processing started for {self.video_path} at {start_time.strftime('%Y-%m-%d %H:%M:%S ')}")

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            end_time = datetime.now()
            elapsed_seconds = int((end_time - start_time).total_seconds())
            minutes, seconds = divmod(elapsed_seconds, 60)
            print(f"Video processing ended for {self.video_path} at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Video processing duration for {self.video_path}: {minutes}m {seconds}s")
            self.manager.remove_video(self.video_path)
            return

        # Get FPS from video
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            self.frame_count += 1
            frame = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)


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
                classes=self.allowed_class_ids,
                tracker="botsort.yaml",
                verbose=False
            )

            for result in results:

                for box in result.boxes:

                    cls = int(box.cls[0])

                    if cls not in [0, 2, 5, 7, 15, 16]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = self.model.names[cls]
                    confidence = float(box.conf[0])
                    crop = frame[y1:y2, x1:x2]

                    if not self.matches_user_request(label, crop):
                        continue

                    object_id = f"{label}_{self.frame_count:04d}"

                    # Get tracking ID from botsort
                    if box.id is not None:
                        track_id = int(box.id[0])
                        object_id = f"ID: {track_id}"

                    self.unique_ids.add(object_id)

                    # Create or update tracked object
                    if object_id not in self.tracked_objects:
                        # Create new object entry
                        if crop.size == 0:
                             continue
                        
                        h, w = crop.shape[:2]
                        if h == 0 or w == 0:
                            continue

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
        
        end_time = datetime.now()
        elapsed_seconds = int((end_time - start_time).total_seconds())
        minutes, seconds = divmod(elapsed_seconds, 60)
        print(f"Video processing ended for {self.video_path} at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Video processing duration for {self.video_path}: {minutes}m {seconds}s")

        # Remove this video from manager
        self.manager.remove_video(self.video_path)