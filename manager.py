import threading
import torch
from ultralytics import RTDETR
from detector import RTDETRVideoProcessor


class VideoManager:

    def __init__(self, model_path):

        #Device Selection
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("=" * 50)
        print(f"Running on: {self.device}")
        if self.device.startswith("cuda"):
            print(f"GPU Name : {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version : {torch.version.cuda}")
        else:
            print("GPU not available. Using CPU.")

        print("=" * 50)
        # Shared model
        self.model = RTDETR(model_path)

        # Active video container
        self.video_container = {}

        self.lock = threading.Lock()

    def add_video(self, video_path):

        processor = RTDETRVideoProcessor(
            model=self.model,
            video_path=video_path,
            manager=self
        )

        thread = threading.Thread(
            target=processor.process_video,
            daemon=True
        )

        with self.lock:
            self.video_container[video_path] = {
                "processor": processor,
                "thread": thread
            }

        thread.start()

        print(f"Added : {video_path}")

    def remove_video(self, video_path):

        with self.lock:

            if video_path in self.video_container:
                del self.video_container[video_path]
                print(f"Removed : {video_path}")

    def get_active_video_count(self):

        with self.lock:
            return len(self.video_container)

    def start(self):
        pass