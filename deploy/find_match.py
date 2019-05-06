import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

from utils import label_map_util
from tensorflow.python.client import timeline
#from utils import visualization_utils as vis_util
import time
#MODEL_NAME = 'fine_tuned_model_19k'
MODEL_NAME = 'fine_tuned_model_ssdLite'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join(MODEL_NAME , 'object-detection.pbtxt')

NUM_CLASSES = 8

graph = tf.Graph()
category_index = {}
def init_model(model_name, num_classes):
	global category_index
	global PATH_TO_FROZEN_GRAPH
	MODEL_NAME = model_name
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
	PATH_TO_LABELS = os.path.join(MODEL_NAME , 'object-detection.pbtxt')
	NUM_CLASSES = num_classes
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	#print(category_index)

def get_graph():
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      print([n.name + '=>' +  n.op for n in od_graph_def.node if n.op in
      ('Softmax','Placeholder')])
      tf.import_graph_def(od_graph_def, name='')
      #print(n.name)
  return detection_graph

def get_categories():
  return category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.4), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      i = 0;
     
        # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
     
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

#      score = output_dict['detection_scores'][0]
#      if score > 0.80:
#        detect_class = output_dict['detection_classes'][0].astype(np.uint8)
#        if detect_class in category_index.keys():
#          detect_class_name = category_index[detect_class]['name']
#        print(output_dict['detection_scores'][0], end = ": ") 
#        print(detect_class_name)
#      else:
#        print("unknown")
      #print(category_index(output_dict['detection_classes'][0].astype(np.uint8)))
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
#
#image_np = load_image_into_numpy_array(Image.open('/home/ubuntu/dd/bin/ed51d73523431568dd3eef166ef15a65_0_70097523720500_0.jpg'))
#detection_graph = get_graph()
#output_dict = run_inference_for_single_image(image_np, detection_graph)

