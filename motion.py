import cv2
import numpy as np
import collections
import imutils
import time

class motion:
    def __init__(self, lumareas, lumdrawarea, imresizewidth, crop, imcrop):
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
        self.lumareas = lumareas #[[90,180,115,250],[750,215,785,300],[300,150,330,176]]
        self.lumareasshow = lumdrawarea
        self.imresizewidth = imresizewidth#800        
        self.crop = crop
        self.imcrop = imcrop#[0, 200, 800, 600]
        
    def run(self, color_image_src):
        
            result = 0
            stats = {}
            
            color_image_src = imutils.resize(color_image_src, width=min(self.imresizewidth, color_image_src.shape[1]))
            if self.crop > 0:
                color_image_src = color_image_src[self.imcrop[1]:self.imcrop[3], self.imcrop[0]:self.imcrop[2]]
         
            #------------------------------------------
            # First Run
            #------------------------------------------
            if self.firstrun == 1:
                width, height = color_image_src.shape[:2]
                self.surface = width * height #Surface area of the image

                self.grey_image = np.zeros(color_image_src.shape, dtype = "uint8")
                self.moving_average = np.zeros(color_image_src.shape, dtype = "float32")

                self.fps_start = time.time()

                self.firstrun = 0
            #------------------------------------------
            
            fps_diff = time.time() - self.fps_start

            self.fps_start = time.time()            

            color_image = cv2.GaussianBlur(color_image_src,(5,5),0)
            
            #------------------------------------------
            # Luminosité
            #------------------------------------------
            lumavg = 0
            for area in self.lumareas:
                if self.lumareasshow > 0:
                    color_image_src = cv2.rectangle(color_image_src,(area[0],area[1]),(area[2],area[3]),(0,0,255),2)
                lumimage = color_image_src[area[1]:area[3], area[0]:area[2]]
                lumimagegrey = cv2.cvtColor(lumimage, cv2.COLOR_RGB2GRAY)
                lumimagegrey_rows, lumimagegrey_cols = lumimagegrey.shape
                lumavgimage = lumimagegrey.sum()/(lumimagegrey_rows*lumimagegrey_cols)
                lumavg += lumavgimage
            
            lumavg = lumavg / len(self.lumareas)
            
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
            stats['LumAVG'] = str(int(lumavgqueueval))
            stats['LumCUR'] = str(int(lumavg))
            stats['LumPVAR'] = str(int(lumpourceevol))
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

            self.cursurface = 0
            # loop over our contours
            for c in cnts:
                area = 0
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
            stats['FrameDuration'] = str(int(fps_diff*1000))
            stats['MotionPAREA'] = str(int(avg*100)/100)
            stats['MotionAREA'] = str(int(self.cursurface))
            stats['MotionCnt'] = str(len(cnts))
            
            if (avg > self.ceil or self.cursurface > 1000) and (lumpourceevol < luminosite_seuil):
                result = 1

            #------------------------------------------
            #Visualisation du diff en rouge sur l'image principal
            #------------------------------------------
            #mask_inv = cv2.bitwise_not(grey_image)
            #img1_bg = cv2.bitwise_and(color_image_src,color_image_src,mask = mask_inv)
            #mask = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2RGB)
            #mask[np.where((mask==[255,255,255]).all(axis=2))] = [0,0,255]
            #color_image_src = cv2.add(img1_bg,mask)
            #------------------------------------------
            
            return result, color_image_src, grey_image, stats
