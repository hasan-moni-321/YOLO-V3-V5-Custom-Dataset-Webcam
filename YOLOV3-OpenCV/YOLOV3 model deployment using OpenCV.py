# Loading necessary library
import cv2 as cv
import numpy as np

# Declaring some variable
cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3


# Reading names of the Object
classNamePath = "coco.names"
className = []
with open(classNamePath, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')
print("Name of the classes :\n",className)
print('Total number of classes are :\n', len(className))

#### Leading model and weight
modelConfiguration = 'yolov3.cfg'
modelWeight = 'yolov3.weights'
#modelConfiguration = 'yolov3-tiny.cfg'
#modelWeight = 'yolov3-tiny.weights'

# Connecting model and weight
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Finding position of the object
def findObject(outPuts, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outPuts:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]* wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold )
    #print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        cv.putText(img, f'{className[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)


# Accessing webcam for image
while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print(net.getUnconnectedOutLayers())

    outPuts = net.forward(outputNames)
    #print(outPuts[0].shape)
    #print(outPuts[1].shape)
    #print(outPuts[2].shape)
    #print(outPuts[0][0])

    findObject(outPuts, img)


    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

