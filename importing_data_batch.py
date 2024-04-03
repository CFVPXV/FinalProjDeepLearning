import os
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import cv2
import json

final_json = {"images": [], "annotations": [], "categories": [
    {"id": 0, "name": "unknown"}, 
    {"id": 1, "name": "vehicle"}, 
    {"id": 2, "name": "pedestrian"}, 
    {"id": 3, "name": 'sign'}, 
    {'id': 4, 'name': 'cyclist'}
    ]}

# Convert the xy center to xy upper left, and return in a coco friendly format!
def convert_xy(center_x, center_y, width, length):
    upper_left_x = (center_x - (0.5 * width))
    upper_left_y = (center_y - (0.5 * length))
    return [upper_left_x, upper_left_y, width, length]

def add_annotation(frame, curr_iter):
    for i in frame.camera_labels:
        for j in i.labels:
            final_json["annotations"].append({"image_id": curr_iter, "bbox": convert_xy(j.box.center_x, j.box.center_y, j.box.width, j.box.length), "category_id": j.type})


def split_sets(set_data, set_name):

    for cmd in set_data:
        os.system(f"gsutil -m cp {cmd} ./intermediary")

    #os.system("rm intermediary/{cmd}")
    
        inter = os.listdir("intermediary")
        dataset = tf.data.TFRecordDataset(f"intermediary/{inter[0]}", compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break

        for index, image in enumerate(frame.images):
            filename = f"img{item}.jpg"
            final_json["images"].append({"id": item, "file_name": filename})
            add_annotation(frame, curr_iter=item)
            cv2.imwrite(f"{set_name}/{filename}", np.array(tf.image.decode_jpeg(image.image)))
            item += 1

        os.system(f"rm intermediary/{inter[0]}")

    with open(f"{set_name}/labels.json", "w") as outfile:
        json.dump(final_json, outfile)


item = 0

with open("ingestion.txt", "r") as file:
    all_data = file.readlines()
    for i in range(len(all_data)):
        all_data[i] = all_data[i].strip()


print(all_data)

train = all_data[:len(all_data) * 0.8]

valid = all_data[:len(all_data) * 0.8:]

split_sets(train, "train")

split_sets(valid, "valid")
