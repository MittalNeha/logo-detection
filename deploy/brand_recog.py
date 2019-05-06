#import redis
import sys
print(sys.path)
sys.path.append("../")
from PIL import Image
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
import scipy.misc

#from utils import label_map_util
#from find_match import run_inference_for_single_image
import find_match as ssd
from utils import visualization_utils as vis_util


import numpy as np
import logging
import glob

LOG_FILENAME = 'ssd_log.out'

logger = logging.getLogger('SSD')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

hand = logging.FileHandler(LOG_FILENAME, "a", encoding=None, delay="true")
hand.setLevel(logging.DEBUG)
hand.setFormatter(formatter)
logger.addHandler(hand)

#read all files froma folder
FOLDER = '/data_partition/Machine_learning/object-detection/server_images/tested_images/rcnn_7k/FP'
#FOLDER = '/data_partition/VideoDetection/Issues/W4/try/copy'
filelist = glob.glob('/data_partition/Machine_learning/object-detection/server_images/tested_images/rcnn_7k/FP/*.jpg')
#running = True
#while running:
idx = 0
file_name = 'img'
def draw_rect(image):
    global idx
    detect_class_name = []
    #(width, height) = image.size
    output_dict = ssd.run_inference_for_single_image(image, detection_graph)
    
    
#    print(file_name, end = ": ")
#    logger.debug('{}: '.format(file_name))
    for i in range (0,5):
        score = output_dict['detection_scores'][i]
        print(score, output_dict['detection_classes'][i])
        if score > 0.50:
            detect_class = output_dict['detection_classes'][i].astype(np.uint8)
            category_index = ssd.get_categories()
            #print(category_index)
            if detect_class in category_index.keys():
                    detect_class_name.append( category_index[detect_class]['name'] )
                    #logger.debug('{}:{} : {}'.format(file_name, score, detect_class_name)) 
                    print('{}: {}, {}'.format(score, detect_class_name, os.path.basename(file_name)))
                                        
    return detect_class_name

def init_detection():
    global detection_graph
    #ssd.init_model('fine_tuned_model_ssdLite', 7)
    ssd.init_model('fine_tuned_model_resnet', 7)
    category_index = ssd.get_categories()
    print(category_index)
    detection_graph = ssd.get_graph()

def run_detection(filename):
    image = Image.open('images/' + filename)
    detected = draw_rect(image)
    print(detected)
    return detected
