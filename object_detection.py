import coco_dataset
import rickshaw_data
from pathlib import Path
import subprocess
import os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

PATH_CSVFILE = Path(__file__).resolve().parent / "images" / "train" / "train_labels.csv"
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR /"darknet" / "build" / "darknet" / "x64" / "data"
OBJ_PATH = DATA_PATH/ "obj"
ANNOTS_FILE = DATA_PATH / "train.txt"
DARKNET_PATH = ROOT_DIR / "darknet"

classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

# If the annotaion file already exists then delete
if os.path.exists(ANNOTS_FILE):
    os.remove(ANNOTS_FILE)

# Format and apply Rickshaw annotations
csvdata = rickshaw_data.set_darknet_format(PATH_CSVFILE)
rickshaw_data.apply_annotations(csvdata)

# Download COCO data and apply annotations
images = coco_dataset.download_data(OBJ_PATH , classes)
logging.info(images[0])
coco_dataset.apply_annotations(ANNOTS_FILE, classes, images)
os.chdir(DARKNET_PATH)
logging.info(os.getcwd())

# Train DARKNET!
subprocess.run("./darknet detector train ./build/darknet/x64/data/obj.data ./cfg/yolo-obj.cfg ./build/darknet/x64/yolov4.conv.137", capture_output=True)

#Perform Detection
subprocess.run("./darknet detector test ./build/darknet/x64/data/obj.data ./cfg/yolo-obj.cfg ./backup/yolo-obj_best.weights")