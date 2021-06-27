#setting up annotation of rickshow dataset
#This file setup all the annotation of rickshow dataset


#Libraries
import os
import glob
from pathlib import Path
import pandas as pd
import shutil
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

NOTEBOOK_PATH = Path(__file__).resolve().parent
DATA_PATH = NOTEBOOK_PATH.parent /"darknet" / "build" / "darknet" / "x64" / "data"
OBJ_PATH = DATA_PATH / "obj"

def set_darknet_format(labels_path):
    """
    Takes file path of the csv file and converts it into dataframe. 
    Perfoms operations to make the annotation in the required form of
    darknet. 

    Parameters
    ----------
    labels_path : pathlib.Path
        Path of the csv file.

    Returns
    -------
    Modified dataframe
    
    """
    df_annot = pd.read_csv(labels_path)

    # ### Finding cx, cy, w, and h
    df_annot["cx"] = (df_annot["xmin"] + df_annot["xmax"])/2
    df_annot["cy"] = (df_annot["ymin"] + df_annot["ymax"])/2
    df_annot["w"] = df_annot["xmax"] - df_annot["xmin"]
    df_annot["h"] = df_annot["ymax"] - df_annot["ymin"]
    df_annot["class"] = 0

    # Scaling center-x, center-y coordinates and width with respect to image resolution
    df_annot["cx"] = (df_annot["cx"]/df_annot["width"]).round(6)
    df_annot["w"] = (df_annot["w"]/df_annot["width"]).round(6)
    df_annot["cy"] = (df_annot["cy"]/df_annot["height"]).round(6)
    df_annot["h"] = (df_annot["h"]/df_annot["height"]).round(6)
    df_annot = df_annot.applymap(str)
    return df_annot

def apply_annotations(df_annot):
    """
    Takes annotation in the form of dataframe, read from the dataframe
    and write them into a text file, for every '.jpg' file named same as
    the '.jpg' file. 
    Checks if a text file already exist, then delete all of text files.
    After deleting files, new text files are generated according to the 
    annotations then

    Parameters
    ----------
    df_annot : Pandas dataframe
        Pandas dataframe returned by set_darknet_format.

    """
    # ### Delete already existing file
    for file in glob.glob(str(OBJ_PATH) + '/*.txt'):
        if os.path.exists(file):
            os.remove(file)

    df = df_annot.set_index("filename")

    for index, row in df.iterrows():
        file_path = OBJ_PATH / (index[:-3] + "txt")
        f= open(file_path, "a+")
        string = row["class"]+ ' '+ row["cx"]+ ' '+row["cy"]+ ' '+row["w"]+ ' '+row["h"] + '\n'
        f.write(string)
        f.close()

    for name in df_annot["filename"].unique():
        traintxt_path = DATA_PATH / "train.txt"
    #   Creating train.txt
        if os.path.isfile(traintxt_path):
            f = open(traintxt_path, "r")
            if string in f.read():
                f.close()
                continue
            f.close()
        t= open(traintxt_path, "a+")
        t.write("./build/darknet/x64/data/obj/" + name + '\n')
        t.close()

    logging.info("Text files of annotation has been created")
    src_dir = NOTEBOOK_PATH / "rikshaw_dataset"
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(jpgfile, OBJ_PATH)
    logging.info("Images copied and text files created")
