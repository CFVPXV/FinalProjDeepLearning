import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

FILENAME = "data/waymotfrecord1"

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

for data in dataset:
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(data.numpy()))
  break

item = 0
for index, image in enumerate(frame.images):
  item += 1
  cv2.imwrite(f"img{item}.jpg", np.array(tf.image.decode_jpeg(image.image)))
  print(item)



#print(frame.camera_labels[0])
