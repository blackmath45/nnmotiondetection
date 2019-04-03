# import the necessary packages
#from __future__ import print_function
#from imutils.object_detection import non_max_suppression
#from imutils import paths
import cv2
import numpy as np
#import argparse
import imutils
#import os
#import urllib
import time
#import copy
#import sys
import datetime
from time import sleep
#import collections
import threading
from multiprocessing import Process, Queue
from queue import Empty
from nn import nn
from motion import motion
import curses
import smtplib
import imghdr
from email.message import EmailMessage

#**************************************************************************************************

def image_display(taskqueue):
    
    excludeobjects = []
    excludeobjects.append("bench")

    t = motion()
    yy = nn("***REMOVED***/frozen_inference_graph.pb", "***REMOVED***/graph.pbtxt", "***REMOVED***/mscoco_label_map.pbtxt", 0.4, 0.4, excludeobjects)
    #cv2.namedWindow("Target", cv2.WINDOW_AUTOSIZE);
    #cv2.namedWindow("NN", cv2.WINDOW_AUTOSIZE);
    
    sendmailqueueimg = []
    
    start_time = time.time()
    last_move = time.time()
    
    while True:
        image = taskqueue.get()              # Added
        
        sendmail = 0
        
        if not (image is None):        
            result, color_image_src, grey_image, stats = t.run(image)

            #------------------------------------------
            # ncurses
            #------------------------------------------ 
            stdscr.clear()

            stdscr.addstr(0, 0, "Frame Duration :")
            stdscr.addstr(0, 20, stats['FrameDuration'])
            
            stdscr.addstr(5, 0, "Queue :")
            stdscr.addstr(5, 20, str(taskqueue.qsize()))

            stdscr.addstr(1, 0, "Luminosité moyenne :")
            stdscr.addstr(1, 21, stats['LumAVG'])           
            stdscr.addstr(1, 30, "Luminosité courante :")
            stdscr.addstr(1, 51, stats['LumCUR'])  
            stdscr.addstr(1, 60, "% variation :")
            stdscr.addstr(1, 81, stats['LumPVAR'])  
 
            stdscr.addstr(2, 0, "% surface totale :")
            stdscr.addstr(2, 21, stats['MotionPAREA'])           
            stdscr.addstr(2, 30, "Surface :")
            stdscr.addstr(2, 51, stats['MotionAREA'])  
            stdscr.addstr(2, 60, "Nb contours :")
            stdscr.addstr(2, 81, stats['MotionCnt'])  

            stdscr.refresh()    
            #------------------------------------------ 
            
            #cv2.putText(color_image_src,"q:" + str(taskqueue.qsize()),(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)         
            #cv2.imshow("Target", color_image_src)

            if result > 0:

                elapsed_time = time.time() - start_time
                
                imagenn = image.copy()
                imagenn = imagenn[680:1335, 600:1600] #from 600;680 to 1935;1335
                objectsdetected = yy.run(imagenn)
                #imagenn_small = imutils.resize(imagenn, width=min(800, imagenn.shape[1]))
                #cv2.imshow("NN", imagenn_small)
                
                isInteresting = 0
                
                for obj in objectsdetected:
                    if (obj[0] == 'person') or (obj[0] == 'car') or (obj[0] == 'bicycle') or (obj[0] == 'motorcycle') or (obj[0] == 'bus') or (obj[0] == 'truck'):
                        isInteresting = 1
                        
                if (elapsed_time > 1) and (isInteresting == 1):
                    now = datetime.datetime.now()
                    image = image[680:1335, 320:1935]
                    cv2.imwrite('***REMOVED***/cap/IM' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(color_image_src))
                    cv2.imwrite('***REMOVED***/cap/IMFull' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(image))
                    cv2.imwrite('***REMOVED***/cap/IMIA' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(imagenn))
                    
                    last_move = time.time()
                    ret, tmpimg = cv2.imencode(".jpg", np.asarray(image))
                    sendmailqueueimg.append([tmpimg,now.strftime('%Y-%m-%d-%H-%M-%S')])                

                    start_time = time.time()
                    
            else :
                isInteresting = 0

            #------------------------------------------
            # MAIL
            #------------------------------------------   
            size = 0
            for img in sendmailqueueimg:
                size += len(img[0])
                if size > 10000000:
                    sendmail = 1
                    
            last_move_delta = time.time() - last_move
            
            if (last_move_delta > 30) and (len(sendmailqueueimg) > 0) :
                sendmail = 1

            if sendmail == 1:
                print("Sending")
                msg = EmailMessage()
                msg['Subject'] = 'Motion Detection'
                msg['From'] = '***REMOVED***'
                msg['To'] = '***REMOVED***'
                msg.preamble = 'Motion Detection'

                for file in sendmailqueueimg:
                    img_data = file[0].tobytes()
                    msg.add_attachment(img_data, maintype='image', subtype=imghdr.what(None, img_data), filename=file[1])

                # Send the email via our own SMTP server.
                with smtplib.SMTP('***REMOVED***') as s:
                    s.login("***REMOVED***", "***REMOVED***")
                    s.send_message(msg)
                    print("Message sent")

                sendmail = 0
                sendmailqueueimg[:] = []
                
            #------------------------------------------ 

            c = stdscr.getch()
            if c == ord('q'):
                break  # Exit the while()

            #c = cv2.waitKey(7) % 0x100
            #if c == 27 or c == 10:
            #    break
            
    cv2.destroyAllWindows()
    print("Exiting process motion")


if __name__=="__main__":
    
    print("Starting")
    
    #------------------------------------------
    # Init ncurses
    #------------------------------------------    
    stdscr = curses.initscr()
    curses.noecho()
    stdscr.timeout(0)
    #------------------------------------------    
    
    taskqueue = Queue()
    capture = cv2.VideoCapture("rtsp://***REMOVED***")
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
    p.join() 

    #------------------------------------------
    # End ncurses
    #------------------------------------------
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()    
    #------------------------------------------    
