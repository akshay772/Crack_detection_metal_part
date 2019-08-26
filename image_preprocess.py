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
DATA_FOLDER = sys.argv[1]
DEST_FOLDER = sys.argv[2]
# DATA_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1'
# DEST_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1_metal'

if inputFilePath is None or outputFilePath is None:
  print("Bad parameters. Please specify input file path and output file path")
  exit()

count = 1
os.chdir(DATA_FOLDER)
for filename in glob.glob("*.jpg"):
    print(filename)

    current_dir =  os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(current_dir, DATA_FOLDER, filename)
    # print(filepath)
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
    cv2.imwrite(DEST_FOLDER + '/' + filename, gray)
    count += 1
# new_im = Image.fromarray(gray)
# new_im.show()
