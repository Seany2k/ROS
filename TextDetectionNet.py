from OpenVINO import OpenVINO
import cv2
import numpy as np

class TextDetectionNet(OpenVINO):
    def __init__(self, modelDir, modelName = "text-detection-0003"):
        super(TextDetectionNet, self).__init__(modelDir, modelName)
        print(self.net)

    def predict(self, inputs):
        image = inputs
        outputs = []
        out_blob = cv2.dnn.blobFromImage(
            image,
            scalefactor = 1.0,
            size = image.shape[:2][::-1],
            mean = (0,0,0),
            swapRB = False,
            crop = False
        )
        self.net.setInput(out_blob)
        res = self.net.forward()
        print(res.shape)
        # for row in res[0]:
            # cv2.imshow("frame", row)
            # cv2.waitKey(0)
        return res[0][1]

net = TextDetectionNet("/opt/intel/openvino_2020.3.341/deployment_tools/open_model_zoo/tools/downloader/intel/text-detection-0003/FP32/", "text-detection-0003")
img1 = cv2.imread("/home/seany/Pictures/img1.png")
mask = net.predict(img1)
cv2.imshow("mask1", mask)
cv2.waitKey(0)


mask = cv2.medianBlur(mask, 5)
mask = cv2.dilate(mask, (4,2), iterations = 1)
# mask = cv2.erode(mask, (4,1), iterations = 1)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5,1), iterations = 0)
thresh = 0.5
mask[mask > thresh] = 1
mask[mask <= thresh] = 0
print(mask)

cv2.imshow("mask2", mask)
cv2.waitKey(0)

mask = cv2.resize(mask, img1.shape[:2][::-1])
print(mask.shape)

cv2.imshow("mask", mask)
cv2.waitKey(0)

mask = np.array(mask, dtype=np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    #cv2.drawContours(img1, [contour], 0, (0, 0, 255), 2)
    cv2.rectangle(img1, (x,y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("mask", np.array(mask, dtype=float))
cv2.waitKey(0)
cv2.imshow("result", img1)
cv2.waitKey(0)