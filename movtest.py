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
import argparse


#****************************************************************************************************************************************************************
def mov_analysis(imagequeue, imagennqueue, brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea):

    t = motion(brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea)

    start_time = time.time()

    while True:
        if imagequeue.qsize() > 30:
            print("Warning : imagequeue : " + str(imagequeue.qsize()))

        image = imagequeue.get()              # Added

        if not (image is None):        
            result, color_image_src, grey_image, stats = t.run(image)

            if result > 0:
                imagennqueue.put([time.time(), image])

    print("Exiting process motion")

#****************************************************************************************************************************************************************
def nn_analysis(imagennqueue, mailqueue, nnpath, nnareas_crop, nnareas_targetarea, detection_crop, detection_targetarea):

    excludeobjects = []
    excludeobjects.append("bench")

    yy = nn(nnpath['frozeninferencegraph'], nnpath['frozeninferenceconfig'], nnpath['labelmap'], 0.4, 0.4, excludeobjects)

    last_time = time.time()
    isInteresting = 0

    while True:

        if imagennqueue.qsize() > 30:
            print("Warning imagennqueue : " + str(imagennqueue.qsize()))

        datann = imagennqueue.get()
        if not (datann is None):
            imagetime = datann[0]
            image = datann[1]
            imagenn = image.copy()

            elapsed_time = imagetime - last_time

            if (elapsed_time > 30):
                isInteresting = 0

            if nnareas_crop > 0:
                imagenn = imagenn[nnareas_targetarea[1]:nnareas_targetarea[3], nnareas_targetarea[0]:nnareas_targetarea[2]]#[680:1335, 600:1600] #from 600;680 to 1935;1335

            #imagenn_small = imutils.resize(imagenn, width=min(800, imagenn.shape[1]))
            objectsdetected = yy.run(imagenn)
            
            for obj in objectsdetected:
                if (obj[0] == 'person') or (obj[0] == 'car') or (obj[0] == 'bicycle') or (obj[0] == 'motorcycle') or (obj[0] == 'bus') or (obj[0] == 'truck') :
                    isInteresting = 1
                    
            if (elapsed_time > 1) and (isInteresting == 1):
                if detection_crop > 0:
                    image = image[detection_targetarea[1]:detection_targetarea[3], detection_targetarea[0]:detection_targetarea[2]]#[680:1335, 320:1935]

                now = datetime.datetime.now()
                ret, tmpimg = cv2.imencode(".jpg", np.asarray(imagenn)) #mettre image 
                mailqueue.put([tmpimg,now.strftime('%Y-%m-%d-%H-%M-%S')])

                last_time = imagetime

    print("Exiting process nn")

#****************************************************************************************************************************************************************
def mail_sending(mailqueue, detection_mail):

    sendmailqueueimg = []
    sendmailpacket = []

    sendmail = 0
    last_move = time.time()

    while True:

        if mailqueue.qsize() > 30:
            print("Warning mailqueue : " + str(mailqueue.qsize()))

        if ((mailqueue.qsize() > 0) and (sendmail == 0)):
            data = mailqueue.get()
            if not (data is None):
                last_move = time.time()
                sendmailqueueimg.append(data)
        else:
            time.sleep(1)

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

            try:
                with smtplib.SMTP(detection_mail['mail_SMTP']) as s:
                    s.login(detection_mail['mail_login'], detection_mail['mail_password'])
                    s.send_message(msg)
                    print("Message sent\n")

                sendmail = 0
                sendmailqueueimg[:] = []
            except:
                print ("Error when sending email\n")
                time.sleep(10)
        #------------------------------------------

    print("Exiting process mail")
        

if __name__=="__main__":
    
    print("Starting")

    #------------------------------------------
    # Arguments
    #------------------------------------------      
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True,
                    help = 'path to config file')
    args = ap.parse_args()
    #------------------------------------------   
    

    #------------------------------------------
    # Load Config
    #------------------------------------------  
    config = ConfigParser()
    config.read(args.config)
    
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

    imagequeue = Queue()
    imagennqueue = Queue()
    mailqueue = Queue()

    capture = cv2.VideoCapture(camera_url)

    p = Process(target=mov_analysis, args=(imagequeue, imagennqueue, brightanalysis_targetarea, brightanalysis_drawarea, motionareas_resize, motionareas_crop, motionareas_targetarea, ))
    p.start()

    nnp = Process(target=nn_analysis, args=(imagennqueue, mailqueue, nnpath, nnareas_crop, nnareas_targetarea, detection_crop, detection_targetarea, ))
    nnp.start()

    mailp = Process(target=mail_sending, args=(mailqueue, detection_mail, ))
    mailp.start()

    cpt = 0
    skipframecnt = 0

    while True:

        ret, color_image_src = capture.read()   
        cpt +=1

        sys.stdout.write('| frame {:4d}; movqueue {:4d}; nnqueue {:4d}; mailqueue :  {:4d}; skip : {:4d}\r'.format(cpt, imagequeue.qsize(), imagennqueue.qsize(), mailqueue.qsize(), skipframecnt))
        sys.stdout.flush()
        
        if cpt > 2:

            if imagequeue.qsize() > 30:
                skipframecnt+=1
            else:
                imagequeue.put(color_image_src)  # Added

            cpt = 0
            
        time.sleep(0.010)     # Added
        if not p.is_alive():
            print("not alive")
            break

    print("Joining")
    capture.release()
    p.join() 
    nnp.join()
    mailp.join()

