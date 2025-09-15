#0701.py
'''
ref: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
'''
import cv2
import numpy as np

#1
with open("data/coco90-2017.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['data/dog.jpg',   'data/person.jpg',
              'data/horses.jpg','data/eagle.jpg']
src = cv2.imread(image_name[1])
height, width, channels = src.shape

#3
#3-1
path = "data/"
net = cv2.dnn.readNet(path + "frozen_inference_graph.pb",
                      path + "faster_rcnn_resnet50_coco_2018_01_28.pbtxt")

#3-2
blob = cv2.dnn.blobFromImage(src, size=(300, 300), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward() # predict, infer, detect, ...
print("outs.shape = ", outs.shape)

##model = cv2.dnn_DetectionModel(net)
##model.setInputParams(size=(300, 300), mean=(0, 0, 0), swapRB=True)
##outs = model.predict(src)[0]

#4
for detection in outs[0,0,:,:]:

    label = int(detection[1])
    score = float(detection[2])

    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)

    if score > 0.5:

        left = int(detection[3] * width)
        top  = int(detection[4] * height)
        right= int(detection[5] * width)
        bottom=int(detection[6] * height)
        cv2.rectangle(src, (left, top), (right, bottom), color, thickness=2)
        cv2.putText(src, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()

