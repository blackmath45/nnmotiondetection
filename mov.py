import cv2
import numpy as np
import imutils
import time
import sys
import datetime
from time import sleep
import threading
from multiprocessing import Process, Queue
from queue import Empty
from nn import nn
from motion import motion
import curses
import smtplib
import imghdr
from email.message import EmailMessage
from configparser import ConfigParser

#**************************************************************************************************

def image_display(taskqueue, nnpath, nnareas_crop, nnareas_targetarea, brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea, detection_crop, detection_targetarea, detection_mail):

    excludeobjects = []
    excludeobjects.append("bench")

    t = motion(brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea)
    yy = nn(nnpath['frozeninferencegraph'], nnpath['frozeninferenceconfig'], nnpath['labelmap'], 0.4, 0.4, excludeobjects)
    
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
            #stdscr.clear()

            #stdscr.addstr(0, 0, "Frame Duration :")
            #stdscr.addstr(0, 20, stats['FrameDuration'])
            
            #stdscr.addstr(5, 0, "Queue :")
            #stdscr.addstr(5, 20, str(taskqueue.qsize()))

            #stdscr.addstr(1, 0, "Luminosité moyenne :")
            #stdscr.addstr(1, 21, stats['LumAVG'])           
            #stdscr.addstr(1, 30, "Luminosité courante :")
            #stdscr.addstr(1, 51, stats['LumCUR'])  
            #stdscr.addstr(1, 60, "% variation :")
            #stdscr.addstr(1, 81, stats['LumPVAR'])  
 
            #stdscr.addstr(2, 0, "% surface totale :")
            #stdscr.addstr(2, 21, stats['MotionPAREA'])           
            #stdscr.addstr(2, 30, "Surface :")
            #stdscr.addstr(2, 51, stats['MotionAREA'])  
            #stdscr.addstr(2, 60, "Nb contours :")
            #stdscr.addstr(2, 81, stats['MotionCnt'])  

            #stdscr.refresh()    
            #------------------------------------------ 
            
            cv2.putText(color_image_src,"q:" + str(taskqueue.qsize()),(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)         
            cv2.imshow("Target", color_image_src)

            if result > 0:

                elapsed_time = time.time() - start_time
                
                imagenn = image.copy()
                if nnareas_crop > 0:
                    imagenn = imagenn[nnareas_targetarea[1]:nnareas_targetarea[3], nnareas_targetarea[0]:nnareas_targetarea[2]]#[680:1335, 600:1600] #from 600;680 to 1935;1335
                objectsdetected = yy.run(imagenn)
                imagenn_small = imutils.resize(imagenn, width=min(800, imagenn.shape[1]))
                cv2.imshow("NN", imagenn_small)
                
                isInteresting = 0
                
                for obj in objectsdetected:
                    if (obj[0] == 'person') or (obj[0] == 'car') or (obj[0] == 'bicycle') or (obj[0] == 'motorcycle') or (obj[0] == 'bus') or (obj[0] == 'truck'):
                        isInteresting = 1
                        
                if (elapsed_time > 1) and (isInteresting == 1):
                    now = datetime.datetime.now()
                    if detection_crop > 0:
                        image = image[detection_targetarea[1]:detection_targetarea[3], detection_targetarea[0]:detection_targetarea[2]]#[680:1335, 320:1935]
                    #cv2.imwrite('/home/blackmath/TEST/cap/IM' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(color_image_src))
                    #cv2.imwrite('/home/blackmath/TEST/cap/IMFull' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(image))
                    #cv2.imwrite('/home/blackmath/TEST/cap/IMIA' + now.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg',np.asarray(imagenn))
                    
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
                msg['Subject'] = detection_mail['mail_Subject']
                msg['From'] = detection_mail['mail_From']
                msg['To'] = detection_mail['mail_To']
                msg.preamble = detection_mail['mail_preamble']

                for file in sendmailqueueimg:
                    img_data = file[0].tobytes()
                    msg.add_attachment(img_data, maintype='image', subtype=imghdr.what(None, img_data), filename=file[1])

                # Send the email via our own SMTP server.
                with smtplib.SMTP(detection_mail['mail_SMTP']) as s:
                    s.login(detection_mail['mail_login'], detection_mail['mail_password'])
                    s.send_message(msg)
                    print("Message sent")

                sendmail = 0
                sendmailqueueimg[:] = []
                
            #------------------------------------------ 

            #c = stdscr.getch()
            #if c == ord('q'):
                #break  # Exit the while()

            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
               break
            
    cv2.destroyAllWindows()
    print("Exiting process motion")


if __name__=="__main__":
    
    print("Starting")
        
    #------------------------------------------
    # Init ncurses
    #------------------------------------------    
    #stdscr = curses.initscr()
    #curses.noecho()
    #stdscr.timeout(0)
    #------------------------------------------    

    #------------------------------------------
    # Load Config
    #------------------------------------------  
    config = ConfigParser()
    config.read('config.ini')
    
    camera_url = config.get('camera', 'url')
    
    nnpath_frozeninferencegraph = config.get('nnpath', 'frozeninferencegraph')
    nnpath_frozeninferenceconfig = config.get('nnpath', 'frozeninferenceconfig')
    nnpath_labelmap = config.get('nnpath', 'labelmap')
    nnpath = {"frozeninferencegraph": nnpath_frozeninferencegraph, "frozeninferenceconfig": nnpath_frozeninferenceconfig, "labelmap" : nnpath_labelmap}
    
    brightanalysis_targetarea = config.get('brightanalysis', 'targetarea')
    brightanalysis_targetarea = [[int(el) for el in line.split(",")] for line in brightanalysis_targetarea.split(";")]
    brightanalysis_drawarea = config.getint('brightanalysis', 'drawarea')    
    
    motionareas_resize = config.getint('motionareas', 'resize')
    motionareas_crop = config.getint('motionareas', 'crop')
    motionareas_targetarea = config.get('motionareas', 'targetarea')
    motionareas_targetarea = [int(line) for line in motionareas_targetarea.split(",")]
    
    nnareas_crop = config.getint('nnareas', 'crop')
    nnareas_targetarea = config.get('nnareas', 'targetarea')
    nnareas_targetarea = [int(line) for line in nnareas_targetarea.split(",")]
    
    detection_crop = config.getint('detection', 'crop')
    detection_targetarea = config.get('detection', 'targetarea')
    detection_targetarea = [int(line) for line in detection_targetarea.split(",")]   

    detection_mail_sendmail = config.getint('detection', 'mail_sendmail')
    detection_mail_Subject = config.get('detection', 'mail_Subject')
    detection_mail_From = config.get('detection', 'mail_From')
    detection_mail_To = config.get('detection', 'mail_To')
    detection_mail_preamble = config.get('detection', 'mail_preamble')
    detection_mail_SMTP = config.get('detection', 'mail_SMTP')
    detection_mail_login = config.get('detection', 'mail_login')
    detection_mail_password = config.get('detection', 'mail_password')
    detection_mail = {"mail_sendmail": detection_mail_sendmail, 
                      "mail_Subject": detection_mail_Subject, 
                      "mail_From": detection_mail_From, 
                      "mail_To": detection_mail_To, 
                      "mail_preamble": detection_mail_preamble, 
                      "mail_SMTP": detection_mail_SMTP, 
                      "mail_login": detection_mail_login, 
                      "mail_password" : detection_mail_password}
    #------------------------------------------  

    taskqueue = Queue()
    
    capture = cv2.VideoCapture(camera_url)
    
    p = Process(target=image_display, args=(taskqueue,nnpath, nnareas_crop, nnareas_targetarea, brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea, detection_crop, detection_targetarea, detection_mail, ))
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
    #curses.nocbreak()
    #stdscr.keypad(False)
    #curses.echo()
    #curses.endwin()    
    #------------------------------------------    
