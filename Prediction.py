from collections import deque
import time
import subprocess
import numpy as np
import cv2

from EmailAlert import send_Alert

### Prediction
## Read all the frames, perform preprocessing, and perform prediction.

def predict_video(MAX_SEQ_LENGTH, vid_path, vid_out_path, convlstm_model, email):
    video_reader = cv2.VideoCapture(vid_path)
    IMG_SIZE = 120
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_mask = np.ones(shape=(MAX_SEQ_LENGTH), dtype="bool")

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(vid_out_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = MAX_SEQ_LENGTH)

    frame_rate = 15
    prev = 0
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_label = ''

    # Iterate until the video is accessed successfully.
    email_sent = False
    acc_frame_count = 0
    while video_reader.isOpened():

        # time_elapsed = time.time() - prev
        cv2.waitKey(100)
        # Read the frame.
        ok, frame = video_reader.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break
    #     cv2.imwrite('frame.jpg', frame)
    #     out = subprocess.run("/home/suleman/VG_AlexeyAB_darknet/darknet detector test /home/suleman/VG_AlexeyAB_darknet/build/darknet/x64/data/obj.data /home/suleman/VG_AlexeyAB_darknet/cfg/yolo-obj.cfg /home/suleman/VG_AlexeyAB_darknet/weights/yolo-obj_last.weights /home/suleman/Accident_detection/frame.jpg -dont_show", shell=True)
    #     pred_frame = cv2.imread(frame)
        # if time_elapsed > 1./frame_rate:
        #   prev = time.time()
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == MAX_SEQ_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = convlstm_model.predict([np.expand_dims(frames_queue, axis = 0), np.expand_dims(frames_mask, axis = 0)])[0]

            if predicted_labels_probabilities < 0.5:
                acc_frame_count += 1
            if acc_frame_count >=5 and not email_sent:
                email_sent = True
                send_Alert(email)
                predicted_label = f'Accident ({predicted_labels_probabilities})'
            else:
                predicted_label = f'NotAccident ({predicted_labels_probabilities})'

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)
        cv2.imshow('Prediction', frame)

    # Release the VideoCapture and VideoWriter objects.

    video_reader.release()
    video_writer.release()
