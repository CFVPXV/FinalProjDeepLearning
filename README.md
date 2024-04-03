# Final Project Deep Learning

## Description

Following the Waymo dataset, we seek to do two dimensional image classification using Google's MediaPipe

## Steps

Using importing_data_batch.py we will read out data from the google storage bin, and encode into a Coco
format in accordance with Mediapipe...

Then using doingThePipe.py, we will train and export a tflite model via the MediaPipe model.

Finally, moving forward with the tflite model, a simple web app will be built for demonstration and hosted on a Firebase server.
