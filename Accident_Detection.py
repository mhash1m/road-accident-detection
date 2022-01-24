import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from preprocessing import preprocess_frames_in_dir, augment_horrizontal_flip
from Visualize import display_images_in_dir
from Prediction import predict_video

### Sort out paths and variables

train_data = 'train_data'
val_data = 'val_data'
test_data = 'test_data'

MAX_SEQ_LENGTH = 30
IMG_SIZE = 120
IMG_CHANNELS = 1

train_accident_dir = os.path.join(train_data, 'Accident')
train_notAccident_dir = os.path.join(train_data, 'NotAccident')

test_accident_dir = os.path.join(test_data, 'Accident')
test_notAccident_dir = os.path.join(test_data, 'NotAccident')

val_accident_dir = os.path.join(val_data, 'Accident')
val_notAccident_dir = os.path.join(val_data, 'NotAccident')

num_samples_train_accident = len(os.listdir(train_accident_dir))
num_samples_train_notAccident = len(os.listdir(train_notAccident_dir))
num_samples_train = num_samples_train_accident + num_samples_train_notAccident

num_samples_test_accident = len(os.listdir(test_accident_dir))
num_samples_test_notAccident = len(os.listdir(test_notAccident_dir))
num_samples_test = num_samples_test_accident + num_samples_test_notAccident

num_samples_val_accident = len(os.listdir(val_accident_dir))
num_samples_val_notAccident = len(os.listdir(val_notAccident_dir))
num_samples_val = num_samples_val_accident + num_samples_val_notAccident

### Display Images

display_images_in_dir(train_accident_dir)
display_images_in_dir(train_notAccident_dir)

## Preprocessing
# The preprocessing phase involves different techniques to enhance our data, such as:
# Note that we will also need to create masks for video padding ie. to let the model know which frames are padded and which ones are not.
# augment_horrizontal_flip("traindata")

### Preprocessing Traing Data
frame_masks_train = np.zeros(shape=(num_samples_train, MAX_SEQ_LENGTH), dtype="bool")
frame_features_train = np.zeros(
    shape=(num_samples_train, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE), dtype="float32"
)
frame_features, frame_masks = preprocess_frames_in_dir(train_accident_dir, frame_features_train, frame_masks_train, 0)
train_data = preprocess_frames_in_dir(train_notAccident_dir, frame_features_train, frame_masks_train, num_samples_train_accident)


### Preprocessing Validation Data
frame_masks_val = np.zeros(shape=(num_samples_val, MAX_SEQ_LENGTH), dtype="bool")
frame_features_val = np.zeros(
    shape=(num_samples_val, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE), dtype="float32"
)
frame_features, frame_masks = preprocess_frames_in_dir(val_accident_dir, frame_features_val, frame_masks_val, 0)
val_data = preprocess_frames_in_dir(val_notAccident_dir, frame_features_val, frame_masks_val, num_samples_val_accident)

# Now we set our class labels, 0 for Accident and 1 for NotAccident

train_labels = np.append(np.zeros(shape = (num_samples_train_accident)), np.ones(shape = (num_samples_train_notAccident)))
val_labels = np.append(np.zeros(shape = (num_samples_val_accident)), np.ones(shape = (num_samples_val_notAccident)))


### CREATE OUR MODEL
frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
  
#Layers - 1
x = ConvLSTM2D(8, (3, 3), activation = 'tanh', data_format = 'channels_last', recurrent_dropout = 0.2, return_sequences = True)(inputs = frame_features_input, mask = mask_input)
x = BatchNormalization()(x)
x = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same', data_format = 'channels_last')(x)
x = TimeDistributed(Dropout(0.2))(x)
#Layers - 2
x = ConvLSTM2D(16, (3, 3), activation = 'tanh', data_format = 'channels_last', recurrent_dropout = 0.3, return_sequences = True)(x)
x = BatchNormalization()(x)
x = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same', data_format = 'channels_last')(x)
x = TimeDistributed(Dropout(0.4))(x)
#Layers - 3
x = ConvLSTM2D(16, (3, 3), activation = 'tanh', data_format = 'channels_last', recurrent_dropout = 0.35, return_sequences = True)(x)
x = BatchNormalization()(x)
x = MaxPooling3D(pool_size = (1, 2, 2), padding = 'same', data_format = 'channels_last')(x)
x = TimeDistributed(Dropout(0.45))(x)

#Flatten
x = Flatten()(x)
#Dense

output = Dense(1, activation='sigmoid')(x)
convlstm_model = tf.keras.Model([frame_features_input, mask_input], output)
convlstm_model.load_weights("cfg _and_weights/best_weights_AD_val_loss")
convlstm_model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate = 0.001), metrics=['accuracy'])
print(convlstm_model.summary())


## Check Points

log_dir = os.path.join('Logs')
filepath = "best_weights_AD"
tb_callback = TensorBoard(log_dir = log_dir)
# lr_plateau = ReduceLROnPlateau(monitor = "val_loss", patience = 5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_delta=0.04, verbose=1)
cp_best_val_loss = ModelCheckpoint(
      filepath + "_val_loss", monitor='val_loss', mode = 'min', save_weights_only=True, save_best_only=True, verbose=1
)
cp_best_val_acc = ModelCheckpoint(
      filepath + "_val_acc", monitor='val_accuracy', mode = 'max', save_weights_only=True, save_best_only=True, verbose=1
)
cp_last_5 = ModelCheckpoint(
      filepath + "_last_3", save_weights_only=True, save_freq = num_samples_train, verbose=1)


### Train the Model

# convlstm_model_training_history = convlstm_model.fit(x = train_data, y = train_labels, epochs = 500, batch_size = 3, 
#                                                     shuffle = True, validation_data = ([val_data[0], val_data[1]], val_labels), 
#                                                     callbacks = [cp_best_val_loss, cp_best_val_acc, cp_last_5, tb_callback])

### Test on Test Data
## Prepare Test Data 	

frame_masks_test = np.zeros(shape=(num_samples_test, MAX_SEQ_LENGTH), dtype="bool")
frame_features_test = np.zeros(
    shape=(num_samples_test, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE), dtype="float32"
)
frame_features, frame_masks = preprocess_frames_in_dir(test_accident_dir, frame_features_test, frame_masks_test, 0)
test_data = preprocess_frames_in_dir(test_notAccident_dir, frame_features_test, frame_masks_test, num_samples_test_accident)
test_labels = np.append(np.zeros(shape = (num_samples_test_accident)), np.ones(shape = (num_samples_test_notAccident)))

## Evaluate
_, accuracy = convlstm_model.evaluate([test_data[0], test_data[1]], test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")


### PREDICTION
vid_path = 'vid6.mp4'
vid_out_path = 'vid6out.mp4'
alert_email = "example@gmail.com"
predict_video(MAX_SEQ_LENGTH, vid_path, vid_out_path, convlstm_model, alert_email)
