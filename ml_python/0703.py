#0703.py
'''
ref1: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
ref2: https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

'''
import cv2
import numpy as np

#1
with open("./dnn/PreTrained/coco90-2017.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg',   './data/person.jpg',
              './data/horses.jpg','./data/eagle.jpg']
src = cv2.imread(image_name[1])
H, W, C = src.shape

#3
#3-1
path = "./dnn/PreTrained/mask_rcnn_inception_v2_coco_2018_01_28/"
net = cv2.dnn.readNet(path + "frozen_inference_graph.pb",
                      path + "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

#3-2
blob = cv2.dnn.blobFromImage(src, size=(300, 300), swapRB=True, crop=False)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])

print("boxes.shape = ", boxes.shape)
print("masks.shape = ", masks.shape)

#4
dst = src.copy()
for i in range(0, boxes.shape[2]):
    label = int(boxes[0, 0, i, 1])
    score =   boxes[0, 0, i, 2]

    box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
    x1, y1, x2, y2 = box.astype("int")
    boxW = x2 - x1
    boxH = y2 - y1

    color = np.array(COLORS[label%len(COLORS)])

    if score > 0.5:
        clone = src.copy()
        mask = masks[i, label]
        mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.1)
        roi = clone[y1:y2, x1:x2]

        roi = roi[mask]
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
        clone[y1:y2, x1:x2][mask] = blended
        dst[y1:y2, x1:x2][mask] = blended

        color = [int(c) for c in color]
        cv2.rectangle(clone, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(dst,   (x1, y1), (x2, y2), color, 2)

        text = "{}: {:.4f}".format(class_names[label], score)
        cv2.putText(clone, text, (x1, y1 - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(dst, text, (x1, y1 - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Output", clone)
        cv2.waitKey()

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

