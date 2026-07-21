import threading
import time

from manager import VideoManager

manager = VideoManager("Resource\\rtdetr-l.pt")

# Start the manager
manager.start()

# Add videos dynamically
# Example: manager.add_video("VideoFile\\video2.mp4", search_object="person", search_color="red")
# If no object/color is provided, the app will search for all default classes.
manager.add_video("VideoFile\\video2.mp4", search_object="person")
#manager.add_video("VideoFile\\video2.mp4")

time.sleep(5)

# Add another video later
#manager.add_video("VideoFile\\video.mp4")

# Keep the application running
while True:
    time.sleep(1)