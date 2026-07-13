import threading
import time

from manager import VideoManager

manager = VideoManager("Resource\\rtdetr-l.pt")

# Start the manager
manager.start()

# Add videos dynamically
manager.add_video("VideoFile\\video2.mp4")
#manager.add_video("VideoFile\\video2.mp4")

time.sleep(5)

# Add another video later
#manager.add_video("VideoFile\\video.mp4")

# Keep the application running
while True:
    time.sleep(1)