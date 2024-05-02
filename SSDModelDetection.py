######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Adapted from: Evan Juras
# Date: 10/27/19
# Source:
# https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py
# Adapted by: Yu Her
# Adapted date: 5/1/24
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
#
# Notes by Yu Her:
# I have edited the following code by Evan Juras to be suited towards our assignment use case,
# which is to detect a valid TARUMT student ID card through the use of a camera feed.
#
# I have added tesseract OCR code to grab information from the student ID card to be used in validating the card.

# Import packages
import os
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util

import pytesseract
import winsound

import validateID

# set the path for tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(1)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
#                     required=True)
# parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
#                     default='detect.tflite')
# parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
#                     default='labelmap.txt')
# parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
#                     default=0.5)
# parser.add_argument('--resolution',
#                     help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
#                     default='1280x720')
# parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
#                     action='store_true')

# args = parser.parse_args()

def ssd_model_detect(ocr_display, validated_display):
    # set model directory name
    model_name = "SSDModel"
    graph_name = 'detect.tflite'
    labelmap_name = 'labelmap.txt'
    min_conf_threshold = 0.8
    res_w, res_h = 640, 480
    im_w, im_h = int(res_w), int(res_h)
    # use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # Get path to current working directory
    cwd_path = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    path_to_ckpt = os.path.join(cwd_path, model_name, graph_name)

    # Path to label map file
    path_to_labels = os.path.join(cwd_path, model_name, labelmap_name)

    # Load the label map
    with open(path_to_labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del (labels[0])

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=path_to_ckpt)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(im_w, im_h), framerate=30).start()
    time.sleep(1)

    # Create a window
    win_name = 'Phone camera'
    cv2.namedWindow(win_name)

    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # # Resize the frame
        # frame1 = cv2.resize(frame1, (700, 700), interpolation=cv2.INTER_LINEAR)

        # Convert frame to numpy array and get original width and height
        frame = frame1.copy()
        frame_np = np.array(frame)
        original_width, original_height = frame_np.shape[1], frame_np.shape[0]

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
            0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                # Get bounding box coordinates and draw box Interpreter can return coordinates that are outside of
                # image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * im_h)))
                xmin = int(max(1, (boxes[i][1] * im_w)))
                ymax = int(min(im_h, (boxes[i][2] * im_h)))
                xmax = int(min(im_w, (boxes[i][3] * im_w)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

                # Extract the text from the detected object
                object_image = frame[ymin:ymax, xmin:xmax]
                if object_image.size != 0:
                    # convert object_image to grayscale making tesseract work better
                    gray_object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray_object_image)
                    id_check, student_id = validateID.check_id(text)
                    exp_date_check, exp_date = validateID.check_exp_date(text)
                    if id_check and exp_date_check:
                        winsound.Beep(1000, 200)
                        validated_display.set(validateID.get_validated_info(student_id, exp_date))
                    ocr_display.set(text)
                else:
                    print("No object detected in the frame")

        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                    2,
                    cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow(win_name, frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # Press 'q' to quit or the close button
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
