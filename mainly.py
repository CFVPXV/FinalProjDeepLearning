import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import cv2
import os
import json

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

#FILENAME = "data/waymotfrecord1"

# Look for all possible categories later...
final_json = {"images": [], "annotations": [], "categories": []}

files = os.listdir("data/")

#print(files)

item = 0
os.system("mkdir intermediate")

for i in files:
    dataset = tf.data.TFRecordDataset(f"data/{i}", compression_type='')
    for data in dataset:
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(data.numpy()))
      break

    for index, image in enumerate(frame.images):
      item += 1
      filename = f"img{item}.jpg"
      final_json["images"].append({"id": item, "file_name": filename})
      cv2.imwrite(f"intermediate/{filename}", np.array(tf.image.decode_jpeg(image.image)))
      print(item)


with open("intermediate/labels.json", "w") as outfile:
   json.dump(final_json, outfile)
"""
for data in dataset:
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(data.numpy()))
  break

item = 0
for index, image in enumerate(frame.images):
  item += 1
  cv2.imwrite(f"img{item}.jpg", np.array(tf.image.decode_jpeg(image.image)))
  print(item)
"""


#print(frame.camera_labels[0])
