# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import os
import numpy as np
import urllib
import cv2
import time
import copy
import sys
import datetime
from time import sleep
import collections
import threading
from nn import nn
from multiprocessing import Process, Queue
from queue import Empty


print(cv2.__version__)


#**************************************************************************************************


class motion:
    def __init__(self):
        self.ceil = 1
        self.firstrun = 1
        #self.start_time = None
        self.fps_start = None
        self.lumavgqueue = collections.deque(maxlen=100)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.difference = None
        self.grey_image = None
        self.moving_average = None
        self.surface = None
        self.cursurface = 0
        self.temp = None
        
    def run(self, color_image_src):
        
            result = 0
            
            color_image_src = imutils.resize(color_image_src, width=min(800, color_image_src.shape[1]))
            color_image_src = color_image_src[200:600, 0:800]
            
            #------------------------------------------
            # First Run
            #------------------------------------------
            if self.firstrun == 1:
                
                width, height = color_image_src.shape[:2]

                self.surface = width * height #Surface area of the image

                self.grey_image = np.zeros(color_image_src.shape, dtype = "uint8")
                self.moving_average = np.zeros(color_image_src.shape, dtype = "float32")

                #self.start_time = time.time()

                self.fps_start = time.time()

                self.firstrun = 0
            #------------------------------------------
            
            fps_diff = time.time() - self.fps_start

            self.fps_start = time.time()            


            color_image_src = cv2.rectangle(color_image_src,(33,35),(351,59),(0,0,0),-1)
            color_image = cv2.GaussianBlur(color_image_src,(5,5),0)
            
            #------------------------------------------
            # Luminosité
            #------------------------------------------
            lp1_x1 = 90
            lp1_y1 = 180
            lp1_x2 = 115
            lp1_y2 = 250
  
            lp2_x1 = 750
            lp2_y1 = 215
            lp2_x2 = 785
            lp2_y2 = 330  

            lp3_x1 = 300
            lp3_y1 = 150
            lp3_x2 = 330
            lp3_y2 = 176              

            #color_image_src = cv2.rectangle(color_image_src,(lp1_x1,lp1_y1),(lp1_x2,lp1_y2),(0,0,255),2)
            #color_image_src = cv2.rectangle(color_image_src,(lp2_x1,lp2_y1),(lp2_x2,lp2_y2),(0,0,255),2)
            #color_image_src = cv2.rectangle(color_image_src,(lp3_x1,lp3_y1),(lp3_x2,lp3_y2),(0,0,255),2)
            lumimage1 = color_image_src[lp1_y1:lp1_y2, lp1_x1:lp1_x2]
            lumimage2 = color_image_src[lp2_y1:lp2_y2, lp2_x1:lp2_x2]
            lumimage3 = color_image_src[lp3_y1:lp3_y2, lp3_x1:lp3_x2]
            
            lumimagegrey1 = cv2.cvtColor(lumimage1, cv2.COLOR_RGB2GRAY)
            lumimagegrey2 = cv2.cvtColor(lumimage2, cv2.COLOR_RGB2GRAY)
            lumimagegrey3 = cv2.cvtColor(lumimage3, cv2.COLOR_RGB2GRAY)
            
            lumimagegrey_rows1, lumimagegrey_cols1 = lumimagegrey1.shape
            lumimagegrey_rows2, lumimagegrey_cols2 = lumimagegrey2.shape
            lumimagegrey_rows3, lumimagegrey_cols3 = lumimagegrey3.shape
            
            lumavg1 = lumimagegrey1.sum()/(lumimagegrey_rows1*lumimagegrey_cols1)
            lumavg2 = lumimagegrey2.sum()/(lumimagegrey_rows2*lumimagegrey_cols2)
            lumavg3 = lumimagegrey3.sum()/(lumimagegrey_rows3*lumimagegrey_cols3)
            
            lumavg = (lumavg1 + lumavg2 + lumavg3)/3
            
            self.lumavgqueue.append(lumavg)
            lumavgqueueval = 0
            for lu in self.lumavgqueue:
                lumavgqueueval += lu
            lumavgqueueval = lumavgqueueval/len(self.lumavgqueue)
            
            lumpourceevol = abs(100-(100/lumavgqueueval*lumavg))
            
            if lumavgqueueval < 70:
                luminosite_seuil = 20
            else:
                luminosite_seuil = 2			
            
            if (lumpourceevol < luminosite_seuil):
                lumpourceevol_color = (0,255,0)
            else:
                lumpourceevol_color = (0,0,255)
            
            cv2.putText(color_image_src,"avg:" + str(int(lumavgqueueval)),(80,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(color_image_src,"cur:" + str(int(lumavg)),(150,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(color_image_src,"%:" + str(int(lumpourceevol)),(220,30), self.font, 0.5,lumpourceevol_color,1,cv2.LINE_AA)
            #------------------------------------------
            
            if self.difference is None: 
                self.difference = color_image
                self.temp = color_image
                cv2.convertScaleAbs(color_image, self.moving_average, 1.0, 0.0)
            else:
                cv2.accumulateWeighted(color_image, self.moving_average, 0.060, None) #Compute the average 0.020

            cv2.convertScaleAbs(self.moving_average, self.temp, 1.0, 0.0)

            cv2.absdiff(color_image, self.temp, self.difference)

            grey_image = cv2.cvtColor(self.difference, cv2.COLOR_RGB2GRAY)

            _,grey_image = cv2.threshold(grey_image, 30, 255, cv2.THRESH_BINARY)

            grey_image = cv2.dilate(grey_image, None, 18) #to get object blobs
            grey_image = cv2.erode(grey_image, None, 10)
            

            cnts = cv2.findContours(grey_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

            # loop over our contours
            for c in cnts:
                area = 0
                
                # approximate the contour
                #peri = cv2.arcLength(c, True)
                #approx = cv2.approxPolyDP(c, 0.1 * peri, True)
                
                #rect = cv2.minAreaRect(c)
                #box = cv2.boxPoints(rect)
                
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                
                if area > 20:
                    color_image_src = cv2.rectangle(color_image_src,(x,y),(x+w,y+h),(0,255,0),2)
                    self.cursurface += cv2.contourArea(c)

            avg = (self.cursurface*100)/self.surface 
            
            cv2.putText(color_image_src,str(int(fps_diff*1000)),(10,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)			
            cv2.putText(color_image_src,"a:" + str(int(avg*100)/100),(300,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(color_image_src,"s:" + str(int(self.cursurface)),(500,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(color_image_src,"cnt:" + str(len(cnts)),(700,30), self.font, 0.5,(255,255,255),1,cv2.LINE_AA)			
            
            if (avg > self.ceil or self.cursurface > 1000) and (lumpourceevol < luminosite_seuil):
                #print("Something is moving !")
                result = 1
                #elapsed_time = time.time() - self.start_time
                #if elapsed_time > 1:
                #    t = datetime.datetime.now()
                #    cv2.imwrite('/home/blackmath/Motion/cap/ARIM' + t.strftime('%Y-%m-%d-%H-%M%s') + '.png',np.asarray(color_image_src))
                #    cv2.imwrite('/home/blackmath/Motion/cap/ARIMFull' + t.strftime('%Y-%m-%d-%H-%M%s') + '.png',np.asarray(color_image_src_full))
                            
                #    self.start_time = time.time()
            
            self.cursurface = 0

            #------------------------------------------
            #Visualisation du diff en rouge sur l'image principal
            #------------------------------------------
            #mask_inv = cv2.bitwise_not(grey_image)
            #img1_bg = cv2.bitwise_and(color_image_src,color_image_src,mask = mask_inv)
            #mask = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2RGB)
            #mask[np.where((mask==[255,255,255]).all(axis=2))] = [0,0,255]
            #color_image_src = cv2.add(img1_bg,mask)
            #------------------------------------------
            
            return result, color_image_src, grey_image

def image_display(taskqueue):
    
    excludeobjects = []
    excludeobjects.append("bench")

    t = motion()
    yy = nn("/home/blackmath/TEST/frozen_inference_graph.pb", "/home/blackmath/TEST/graph.pbtxt", "/home/blackmath/TEST/mscoco_label_map.pbtxt", 0.4, 0.4, excludeobjects)
    cv2.namedWindow("Target", cv2.WINDOW_AUTOSIZE);
    #cv2.namedWindow("Grey", cv2.WINDOW_AUTOSIZE);
    cv2.namedWindow("NN", cv2.WINDOW_AUTOSIZE);
    start_time = time.time()
    while True:
        image = taskqueue.get()              # Added
        if not (image is None):        
            result, color_image_src, grey_image = t.run(image)
            
            cv2.putText(color_image_src,"q:" + str(taskqueue.qsize()),(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)         
            cv2.imshow("Target", color_image_src)
            
            if result > 0:
                
                elapsed_time = time.time() - start_time
                
                imagenn = image.copy()
                imagenn = imagenn[680:1335, 600:1600] #from 600;680 to 1935;1335
                objectsdetected = yy.run(imagenn)
                imagenn_small = imutils.resize(imagenn, width=min(800, imagenn.shape[1]))
                cv2.imshow("NN", imagenn_small)
                
                isInteresting = 0
                for obj in objectsdetected:
                    if (obj[0] == 'person') or (obj[0] == 'car') or (obj[0] == 'bicycle') or (obj[0] == 'motorcycle') or (obj[0] == 'bus') or (obj[0] == 'truck'):
                        isInteresting = 1
                        
                if (elapsed_time > 1) and (isInteresting == 1):
                    now = datetime.datetime.now()
                    image = image[680:1335, 320:1935]
                    cv2.imwrite('/home/blackmath/TEST/cap/IM' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.png',np.asarray(color_image_src))
                    cv2.imwrite('/home/blackmath/TEST/cap/IMFull' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.png',np.asarray(image))
                    cv2.imwrite('/home/blackmath/TEST/cap/IMIA' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.png',np.asarray(imagenn))
                    
                    start_time = time.time()
                    
            else :
                isInteresting = 0
            
            #cv2.imshow("Grey", grey_image)
            
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break

    cv2.destroyAllWindows()
    print("Exiting process motion")


if __name__=="__main__":
    
    print("Starting")
    taskqueue = Queue()
    capture = cv2.VideoCapture("rtsp://x:x@192.168.0.230")
    p = Process(target=image_display, args=(taskqueue,))
    p.start()
    cpt = 0
    
    while True:

        ret, color_image_src = capture.read()   
        cpt +=1
        
        if cpt > 2:
            taskqueue.put(color_image_src)  # Added
            cpt = 0
            
        time.sleep(0.010)     # Added
        if not p.is_alive():
            print("not alive")
            break

    print("Joining")
    capture.release()
    cv2.destroyAllWindows()
    p.join() 
    
    
