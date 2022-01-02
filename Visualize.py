import os
import random
import matplotlib.pyplot as plt
import cv2

### Display Images
# The following function takes a directory containing videos, and displays middle frames from 20 random videos.
# This is used to display the data and get an idea if our paths and labels are set right.

def display_images_in_dir(dir):
  plt.figure(figsize = (20,20))
  video_folders = os.listdir(dir)
  image_folders_n = random.sample(range(len(video_folders)), 20)
  for i, n in enumerate(image_folders_n):
    image_files = os.listdir(os.path.join(dir, video_folders[n]))
    image = image_files[int(len(image_files)/2)]
    image_path = os.path.join(dir, video_folders[n], image)
    img = cv2.imread(image_path)
    plt.subplot(5, 4, i+1)
    plt.imshow(img)
    plt.axis('off')