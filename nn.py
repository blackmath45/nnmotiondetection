import cv2

class nn:
    def __init__(self,FROZEN_PATH, CONFIG_PATH, LABEL_PATH, confThreshold, nmsThreshold, excludeobjects = []):
        # Load names of classes
        self.classes = {}
        with open(LABEL_PATH, 'rt') as f:
            temp = f.read().replace('\n', '')
            tab = temp.replace('item {','').split('}')
            for line in tab:
                linesplit = line.split(':')
                if len(linesplit) == 4:
                    cid = int(linesplit[2].replace('display_name',''))
                    name = linesplit[3].replace('"','').rstrip().lstrip()
                    self.classes[cid] = name
                    #print(str(cid) + "|" + name)

        self.net = cv2.dnn.readNetFromTensorflow(FROZEN_PATH, CONFIG_PATH)

        self.outNames = self.net.getUnconnectedOutLayersNames()

        self.confThreshold = confThreshold #Confidence threshold
        self.nmsThreshold = nmsThreshold #Non-maximum suppression threshold
        
        self.excludeobjects = excludeobjects
            
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def drawPred(classId, conf, left, top, right, bottom):
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

            label = '%.2f' % conf

            if self.classes:
                label = '%s: %s' % (self.classes[classId+1], label)

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            #cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        layerNames = self.net.getLayerNames()
        lastLayerId = self.net.getLayerId(layerNames[-1])
        lastLayer = self.net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        objectsdetected = []
        if lastLayer.type == 'DetectionOutput':
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            #for out in outs:
                for detection in outs[0, 0]:
                    confidence = detection[2]                  
                    if confidence > self.confThreshold:
                        left = int(detection[3])
                        top = int(detection[4])
                        right = int(detection[5])
                        bottom = int(detection[6])
                        width = right - left + 1
                        height = bottom - top + 1
                        if width * height <= 1:
                            left = int(detection[3] * frameWidth)
                            top = int(detection[4] * frameHeight)
                            right = int(detection[5] * frameWidth)
                            bottom = int(detection[6] * frameHeight)
                            width = right - left + 1
                            height = bottom - top + 1
                        if not (self.classes[int(detection[1])] in self.excludeobjects) :
                            classIds.append(int(detection[1]) - 1)  # Skip background label
                            confidences.append(float(confidence))
                            boxes.append([left, top, width, height])
                            objectsdetected.append([self.classes[int(detection[1])], float(confidence)])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)        
        
        return objectsdetected

    def run(self, image):
        self.net.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
        outs = self.net.forward()
        obj = self.postprocess(image, outs) 
        return obj   
