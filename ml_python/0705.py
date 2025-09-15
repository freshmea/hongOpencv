#0705.py
'''
ref: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
model: MobileNet-SSD v3(weights, config)

'''
import cv2
import numpy as np

#1
with open("data/coco90-2017.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
class_names.insert(0, '')

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg', './data/person.jpg',  './data/horses.jpg']
src = cv2.imread(image_name[0])
H, W, C = src.shape

#3
#3-1
path = "data/"
net = cv2.dnn.readNet(path + "frozen_inference_graph_ssd.pb",
                      path + "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

#3-2
blob = cv2.dnn.blobFromImage(src, size=(320, 320),
                             mean = (127.5, 127.5, 127.5),
                             scalefactor= 1.0 / 127.5,
                             swapRB=True, crop=False)
print("np.min(blob) = ", np.min(blob)) # -1
print("np.max(blob) = ", np.max(blob)) # 1

net.setInput(blob)
outs = net.forward()
print('outs.shape=', outs.shape)

#4:

for detection in outs[0, 0, :, :]:

    label = int(detection[1])
    score = detection[2]
    box =   detection[3:7] * np.array([W, H, W, H])


    x1, y1, x2, y2 = box.astype("int")
    boxW = x2 - x1
    boxH = y2 - y1

    color = COLORS[label%len(COLORS)]

    if score > 0.6:

        cv2.rectangle(src,   (x1, y1), (x2, y2), color, 2)

        text = "{}: {:.4f}".format(class_names[label], score)
        cv2.putText(src, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
