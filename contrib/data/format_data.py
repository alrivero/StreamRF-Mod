import cv2
import os
import shutil
import sys
from pathlib import Path

# Args for where data is located and what the framerate of each video is
data_dir = sys.argv[1]
frame_count = int(sys.argv[2])

# Find all videos within the root
all_videos = []
for file in os.listdir(data_dir):
    if file.endswith(".mp4"):
        all_videos.append(os.path.join(data_dir, file))

# Make directory and give bounds for each frame
for i in range(frame_count):
    new_dir = os.path.join(data_dir, "%04d" % i)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    images_dir = os.path.join(new_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    shutil.copy(os.path.join(data_dir, "poses_bounds.npy"), new_dir)

# Convert each video into a collection of frames
for vid_path in all_videos:
    vidcap = cv2.VideoCapture(vid_path)

    cam_number = int(Path(vid_path).stem.split("_")[1])

    success,image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(data_dir, "%04d" % count, "images", "cam_%02d.png" % cam_number)
        cv2.imwrite(frame_path, image)

        success,image = vidcap.read()
        count += 1