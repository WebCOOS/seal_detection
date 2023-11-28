# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 00:29:08 2023

@author: Fahim
"""

# use "pip install opencv-python" for installation
import cv2
# use "pip install ultralytics" for installation
from ultralytics import YOLO
import time

# Load yolov8 model
model = YOLO('models/best_seal/1/best_seal.pt')

while True:
    try:
        #input stream
        cap = cv2.VideoCapture("https://stage-ams.srv.axds.co/stream/adaptive/noaa/tmmc_prls/hls.m3u8")
        ts = time.time()
        frameid = 0
        ret, frame = cap.read()

        ##Uncomment for saving as .avi
        # dest = str(ts) + ".avi"
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(dest, fourcc, 20.0, (1280, 720))

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                img_boxes = frame

                #use YOLOv8
                results = model.predict(frame, conf = 0.2)
                for result in results:
                    for score, cls, bbox in zip(result.boxes.conf, result.boxes.cls, result.boxes.xyxy):
                        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                        h, w, _ = frame.shape

                        y_min = int(max(1, y1))
                        x_min = int(max(1, x1))
                        y_max = int(min(h, y2))
                        x_max = int(min(w, x2))

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        #label = "Seal" + ": " + ": {:.2f}%".format(score * 100)

                        if cls.item() == 0.0:
                            label = "Rock"+ ": " + ": {:.2f}%".format(score * 100)
                            img_boxes = cv2.rectangle(img_boxes,(x_min, y_max),(x_max, y_min),(0,255,0), 2)
                            cv2.putText(img_boxes, label, (x_min, y_max-10), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                        if cls.item() == 1.0:
                            label = "Seal"+ ": " + ": {:.2f}%".format(score * 100)
                            img_boxes = cv2.rectangle(img_boxes,(x_min, y_max),(x_max, y_min),(0,0,255), 2)
                            cv2.putText(img_boxes, label, (x_min, y_max-10), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

                outp = cv2.resize(img_boxes, (1280, 720))
                ##Uncomment for saving as .avi
                # out.write(outp)
                frameid += 1
                cv2.imshow('PreviewWindow', img_boxes)

                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break
            else:
                break

        cap.release()

        ##Uncomment for saving as .avi
        # out.release()

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        exit(0)
    except:
        pass
