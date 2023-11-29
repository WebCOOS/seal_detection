# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:29:08 2022

@author: Fahim
"""

import numpy
# use "pip install tensorflow" for installation
import tensorflow as tf
# use "pip install opencv-python" for installation
import cv2

import os
import time
import datetime

# Load model
detector = tf.saved_model.load("saved_model/2")
width = 768
height = 768

# detector = tf.saved_model.load("saved_model/3")
# width = 896
# height = 896

ts = time.time()

#Load stream
cap = cv2.VideoCapture("https://stage-ams.srv.axds.co/stream/adaptive/noaa/tmmc_prls/hls.m3u8")

try:
    if not os.path.exists('outputVideos'):
        os.makedirs('outputVideos')

# if not created then raise error
except OSError:
    print('Error: Creating directory of outputVideos')


frameid = 0
ret, frame = cap.read()

dest = "outputVideos/" + str(ts) + ".avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(dest, fourcc, 20.0, (1024, 600))


while cv2.waitKey(1) == -1:
    t1 = datetime.datetime.now()
    ret, frame = cap.read()
    if ret == True:
        inp = cv2.resize(frame, (width , height))

        #Convert img to RGB
        #rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        rgb = inp
        #img_boxes = inp

        #Is optional but i recommend (float convertion and convert img to tensor image)
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

        #Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)
        
        boxes, scores, classes, num_detections = detector(rgb_tensor)
        
        #pred_labels = classes.numpy().astype('int')[0]
        
        #pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
    
    #loop throughout the detections and place a box around it  
        #for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        for score, (ymin,xmin,ymax,xmax) in zip(pred_scores, pred_boxes):
            if score < 0.5:
                continue

            h, w, _ = frame.shape
            
            y_min = int(max(1, (ymin * (h/height))))
            x_min = int(max(1, (xmin * (w/width))))
            y_max = int(min(h, (ymax * (h/height))))
            x_max = int(min(w, (xmax * (w/width))))
                
            score_txt = f'{100 * round(score,0)}'
            img_boxes = cv2.rectangle(frame,(x_min, y_max),(x_max, y_min),(0,255,0),3)      
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "Seal" + ": " + ": {:.2f}%".format(score * 100)
            cv2.putText(img_boxes,label,(x_min, y_max-10), font, 1, (255,0,0), 2, cv2.LINE_AA)
            #cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
        
        outp = cv2.resize(img_boxes, (1024, 600))
        frameid += 1
        out.write(outp)
        t2 = datetime.datetime.now()
        diff=t2-t1
        d = diff.total_seconds()/(60)
        print("processed frame", frameid, "at fps", int(1/d))
        #cv2.imshow('PreviewWindow', outp)
        #print('Showing preview. Click on preview window or press any key to stop.')
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()