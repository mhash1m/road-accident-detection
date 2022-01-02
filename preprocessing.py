import os
import math
import cv2

## Preprocessing
# The preprocessing phase involves different techniques to enhance our data, such as:

# - Data Augmentation
# - Video Padding
# - Convert to Grey Scale
# - Downsizing Images
# - Normalizing Images

def augment_horrizontal_flip(train_path):
    dir_list0 = os.listdir(train_path)
    print(dir_list0)
    for dir0 in dir_list0:
        dir0_path = os.path.join(train_path, dir0)
        dir_list1 = os.listdir(dir0_path)
#         dir_list1 = sorted_alphanumeric(dir_list1)
        for dir1 in dir_list1:
            dir1_path = os.path.join(dir0_path, dir1)
            dir1_path_aug = dir1_path + '_aug'
            if not os.path.isdir(dir1_path_aug):
                os.mkdir(dir1_path_aug)
            img_list = os.listdir(dir1_path)
            for img in img_list:
                img_path = os.path.join(dir1_path, img)
                img_path_aug = os.path.join(dir1_path_aug, img)
                image = cv2.imread(img_path)
                img_aug = cv2.flip(image, 1)
                cv2.imwrite(img_path_aug, img_aug)


def preprocess_frames_in_dir(dir, features, masks, start_idx):
  video_names = os.listdir(dir)
  for nth_video, video in enumerate(video_names):
    if os.path.isdir(os.path.join(dir, video)):
      video_path = os.path.join(dir, video)
      video_length = len(os.listdir(video_path))
      length = min(MAX_SEQ_LENGTH, video_length)
      skip_window = math.ceil(video_length/MAX_SEQ_LENGTH)
      count = 0
      image_names = os.listdir(video_path)
      # print("SKIP:", skip_window)
      for i in range(skip_window, video_length, skip_window):
        if image_names[i] == 'Icon_':
          continue
        image_path = os.path.join(video_path, image_names[i])
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(image_path)
        resized_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
        normalized_img = resized_img/255
        features[start_idx + nth_video, count,] = normalized_img
        masks[start_idx + nth_video, count] = 1
        count += 1
  return (features, masks)