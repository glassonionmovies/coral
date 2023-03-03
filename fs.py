######## Directional-object-tracking-with-TFLite-and-Edge-TPU #########
# Author: Tahir Mahmood
# Date: 18/7/2020
# This code performs object detection and tracking using a pre-trained Tensor Flow Lite (TFLite) model.
# In addition it can track for each unique object how it is moving through the frame of vision.
# As with my previous project there is a filter built into the code so the type of object to track can be specified.
# For example it can be set to only track all 'people' moving through the frame but ignore other objects such as dogs or bikes.
# Practical use: Automated throughfare or doorway monitoring to under stand flow of 'people' in 
#                each direction or calculate how many 'people' are 'inside' at any one time.

# Credits:
# Main boilerplate for object detection and labelling using TFLite & Edge TPU - Evan Juras
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
# This code was based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# Centroid tracking helper code for object tracking:
# https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking 

import argparse
from centroidtracker import CentroidTracker
import OrderAccuracyUtils
import os
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


import collections
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import svgwrite
import tflite_runtime.interpreter as tflite
import time


#-----------Added for tracking
packaged=set()
seen=set()
collected=''
on_table=''




def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)
        
        
        


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(1)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        #cv2.imshow('Object detector', self.frame)

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
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
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=False,
                   default='/home/mendel/google-coral/example-object-tracker/gstreamer/tflite/MyModels/WT_efficientdet-lite-ing_edgetpu.tflite')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='/home/mendel/google-coral/example-object-tracker/gstreamer/ing_label.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.65)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true', default=True)
parser.add_argument('--sf', help='Minimum confidence threshold for displaying detected objects',
                    default=1)




args = parser.parse_args()

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects ={}
old_objects={}

# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = np.subtract(dict2[key], dict1[key])
   return dict3

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
show_resize_factor=float(args.sf)

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = MODEL_NAME

# Path to label map file
PATH_TO_LABELS = '/home/mendel/google-coral/example-object-tracker/gstreamer/ing_s_label.txt'

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

interpreter = OrderAccuracyUtils.make_interpreter(PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Newly added co-ord stuff
leftcount = 0
rightcount = 0 
obsFrames = 0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    currently_seen=set()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # On the next loop set the value of these objects as old for comparison
    old_objects.update(objects)

    # Grab frame from video stream
    frame1 = videostream.read()
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
#    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
#    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
#    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    
    classes = OrderAccuracyUtils.output_tensor(interpreter, 3)
    boxes = OrderAccuracyUtils.output_tensor(interpreter, 1)
    scores = OrderAccuracyUtils.output_tensor(interpreter, 0)
    #rects variable
    rects =[]
  
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            print(f"Object name is {object_name}")
            
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            box = np.array([xmin,ymin,xmax,ymax])

            rects.append(box.astype("int"))

            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (10, 255, 0), 2)

            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            
            
            #seen.add((object_name, 0))
            #print(f"seen is {seen}")

            
            print(f"label is --- {label}")
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    #update the centroid for the objects
    objects = ct.update(rects)

    # calculate the difference between this and the previous frame
    x = DictDiff(objects,old_objects)
    
    if(len(objects))==0:
        packaged=seen-currently_seen

    
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        seen.add((object_name, objectID))
        currently_seen.add((object_name, objectID))
        packaged=seen-currently_seen
        print(f"seen is {seen}")
        print(f"currently_seen is {currently_seen}")
        print(f"packaged is {packaged}")


        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)         

    # Draw framerate in corner of frame
    #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    
    l=list(packaged)
    l=[(x[0],1) for x in l]
    d={key: 0 for (key, value) in l}
    for i in l:
        d[i[0]]=d[i[0]]+i[1]
    #global collected
    collected=str(d)
    collected_text=' Packaged: '+str(d)

    l=list(currently_seen)
    l=[(x[0],1) for x in l]
    d={key: 0 for (key, value) in l}
    for i in l:
        d[i[0]]=d[i[0]]+i[1]
    
    on_table=str(d)
    on_table_text=' On Table:  '+str(d)

    
    
    
    #cv2.putText(frame,str(packaged),(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    fps_text='FPS: {0:.2f}'.format(frame_rate_calc)
    
    
    display_text=fps_text + collected_text + on_table_text
    print(f"display text is {display_text}")
    
    cv2.setWindowTitle('Object detector',display_text )
    if(show_resize_factor > 1):
        frame = cv2.resize(frame, (int(width*show_resize_factor), int(height*show_resize_factor)))

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    #count number of frames for direction calculation
    obsFrames = obsFrames + 1

    #see what the difference in centroids is after every x frames to determine direction of movement
    #and tally up total number of objects that travelled left or right
    if obsFrames % 30 == 0:
        d = {}
        for k,v in x.items():
            if v[0] > 3: 
                d[k] =  "Left"
                leftcount = leftcount + 1 
            elif v[0]< -3:
                d[k] =  "Right"
                rightcount = rightcount + 1 
            elif v[1]> 3:
                d[k] =  "Up"
            elif v[1]< -3:
                d[k] =  "Down"
            else: 
                d[k] = "Stationary"
                

                
        if bool(d):
            print(d, time.ctime()) # prints the direction of travel (if any) and timestamp
    

    # Press 'q' to quit and give the total tally
    if cv2.waitKey(1) == ord('q'):
        print("Left",leftcount,"Right",rightcount)
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
