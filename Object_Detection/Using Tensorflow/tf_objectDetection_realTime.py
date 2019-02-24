## Real Time Object Detection using Tensorflow
# Prashant Brahmbhatt (https:www.github.com/hashbanger)
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util

import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CPKT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

if not os.path.isfile(MODEL_FILE):
    print("File not present, Downloading...")
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE +  MODEL_FILE, MODEL_FILE)
else:
    print("File Found")

    tar_file = tarfile.open(MODEL_FILE)

# Checking for existence of 'frozen_inference_graph.pb'

for file in tar_file.getmembers(): # The get memebers functions returns the list of the contents of the tar file
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    # We read *pb file using GraphDef and bind the GraphDef to a (default) Graph, 
    # then use a session to run the Graph for computation
    with tf.gfile.GFile(PATH_TO_CPKT, 'rb') as fid:
    #tf.gfile is an abstraction for accessing the filesystem and is documented here. 
    # It is recommended over using plain python API since it provides some level of portability
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS) #Loads label map proto.

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes= NUM_CLASSES, use_display_name= True)
# Loads label map proto and returns categories list compatible with eval.

category_index = label_map_util.create_category_index(categories)
# Creates dictionary of COCO compatible categories keyed by category id.

import cv2 # Importing opencv
cap = cv2.VideoCapture(0) 

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read() 
            #"Frame" will get the next frame in the camera (via "cap").
            # "Ret" will obtain return value from getting the camera frame, either true of false.
            
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each box represents a part of the image where a particular object was detected.

            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Actual detection.

            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True,
            line_thickness=8)
            # Visualization of the results of a detection.
            
            cv2.imshow('object detection', cv2.resize(image_np, (1024, 786)))
            # Opening the detection window

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # Quit by pressing the 'q' key

# de nada!