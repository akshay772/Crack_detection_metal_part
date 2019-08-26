# 1. Import the necessary packages
import numpy as np
import cv2
import urllib
from urllib.request import urlopen
from PIL import Image
import webcolors
import time, os, glob
import sys

# DATA_FOLDER = 'YE358311_Fender_apron/defect1'
DATA_FOLDER_DEFECT_HEALTHY = [sys.argv[1] + '/YE358311_defects/YE358311_Crack_and_Wrinkle_defect', sys.argv[1] + '/YE358311_Healthy']
DEST_FOLDER = [sys.argv[2] + '/train/defect', sys.argv[2] + '/train/normal']
# DATA_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1'
# DEST_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1_metal'

if sys.argv[1] is None or sys.argv[2] is None:
  print("Bad parameters. Please specify input file path and output file path")
  exit()

current_dir =  os.path.abspath(os.path.dirname(__file__))
path = os.path.join(current_dir, DATA_FOLDER_DEFECT_HEALTHY[0])
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.jpg')]
no_defect_files = len(files_txt)

current = 0
for DATA_FOLDER in DATA_FOLDER_DEFECT_HEALTHY:
    print("\n\n\nEntering data folder ", DATA_FOLDER[2:])
    count = 1
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(current_dir, DATA_FOLDER)
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.jpg')]
    for filename in files_txt:
        if count > no_defect_files:
            print("Successfully pre-processsed %d files" % (2*(count-1)))
            break

        current_dir =  os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(current_dir, DATA_FOLDER[2:], filename)
        print("\n\nReading file : ", filename)
        img = cv2.imread(filepath, -1)
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((31,31), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 15)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        width = 512
        height = 512
        dim = (width, height)
        resized = cv2.resize(result_norm, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[current][2:])):
            os.makedirs(os.path.join(current_dir, DEST_FOLDER[current][2:]))
        savepath = os.path.join(current_dir, DEST_FOLDER[current][2:], str(count) + '.jpg')
        print("File saving in ", savepath)
        cv2.imwrite(savepath, gray)
        count += 1
    current  += 1

DEST_FOLDER = [sys.argv[2] + '/test/defect', sys.argv[2] + '/test/normal']
if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[0][2:])):
    os.makedirs(os.path.join(current_dir, DEST_FOLDER[0][2:]))
if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[1][2:])):
    os.makedirs(os.path.join(current_dir, DEST_FOLDER[1][2:]))

    # new_im = Image.fromarray(gray)
    # new_im.show()
