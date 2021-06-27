import logging
import os
from pathlib import Path
from typing import Any
from pycocotools.coco import COCO
import requests

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

MAIN_PATH = Path(__file__).resolve().parent
root_dir = MAIN_PATH.parent

COCO_ANNOT_PATH = MAIN_PATH / "instances_train2017.json"

# Download the instances_train2017.json file from coco dataset website. For a direct link download
# you can use https://drive.google.com/file/d/1Cg-hbtlYdMiMFZ88owFIRSfgpE9WI0av/view?usp=sharing
# and save it in the same folder as this file.

def download_data(download_path : Path, classes : dict):
    """
    Selects and Downloads images which belong to the provided classes. 
    The `instances_train2017.json` is used to extract the IDs and URLs 
    of the images with respect to our classes.

    Parameters
    ----------
    download_path : pathlib.Path
        Path to download selected images.
    classes : dict
        Classes of the images to be downloaded.

    Returns
    -------
    Any

    """

    # Extract Image IDs
    coco = COCO(COCO_ANNOT_PATH)
    catIds = coco.getCatIds(catNms=classes)
    count = 0
    for c in catIds:
        if count>0:
            imgIds2= coco.getImgIds(catIds=c)[:125]
            imgIds+=imgIds2
        else:
            imgIds=coco.getImgIds(catIds=c)[:125]
        count+=1
        logging.info(len(imgIds))
    images = coco.loadImgs(imgIds)

    #Downloading images from coco dataset and storing into darknet's obj folder
    for im in images:
        logging.info(im)
        img_data = requests.get(im['coco_url']).content
        with open(download_path / im['file_name'], 'wb') as handler:
            handler.write(img_data)
    return images



def apply_annotations (annots_path : Path, classes : dict, images : Any):
    """
    Selects and Downloads images which belong to the provided classes. 
    The `instances_train2017.json` is used to extract the IDs and URLs 
    of the images with respect to our classes.

    Parameters
    ----------
    annots_path : pathlib.Path
        Path to a text file to save annotations.
    classes : dict
        Classes of the images to be downloaded.
    images : Any
        Images loaded by `download_data`.

    """
    obj_path = root_dir /"darknet" / "build" / "darknet" / "x64" / "data" / "obj"
    coco = COCO(COCO_ANNOT_PATH)
    catIds = coco.getCatIds(catNms=classes)
    for im in images:
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        file_name = im['file_name']
        string = "./build/darknet/x64/data/obj/" + file_name
        f = open(annots_path, "r")
        if string in f.read():
            f.close()
            continue
        f.close()
        t = open(annots_path, "a+")
        t.write(string + '\n')
        t.close()
        file_path = obj_path / (file_name[:-3] + "txt")
        print(file_path)
        for i in range(len(anns)):
            image_w = im['width']
            image_h = im['height']
            w = round(anns[i]['bbox'][2]/image_w, 6)
            h = round(anns[i]['bbox'][3]/image_h, 6)
            cx = round((anns[i]['bbox'][0] + w/2)/image_w, 6)
            cy = round((anns[i]['bbox'][1] + h/2)/image_h, 6)
            cat = anns[i]['category_id']
            if(anns[i]['category_id']==8):
                cat = 5
            string = str(cat) + ' '+ str(cx)+ ' '+str(cy)+ ' '+str(w)+ ' '+str(h) + '\n'
            if os.path.isfile(file_path):
                fl = open(file_path, "r")
                if string in fl.read():
                    fl.close()
                    continue
                fl.close()
            tl= open(file_path, "a+")
            tl.write(string)
            tl.close()
    print("Annotation has been applied")
