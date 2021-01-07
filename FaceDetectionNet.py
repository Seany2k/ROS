import cv2
from model_classes import OpenVINO


class FaceDetectionNet(OpenVINO):
    def __init__(self, model_dir, model_name="face-detection-adas-0001"):
        super(FaceDetectionNet, self).__init__(model_dir, model_name)

    def predict(self, inputs):
        image = inputs[0]
        outputs = []
        out_blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=image.shape[:2],
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )
        self.net.setInput(out_blob)
        res = self.net.forward()
        for row in res[0][0]:
            if row[0] == -1:
                break
            if row[2] < 0.9:
                continue

            x1 = max(0, int(image.shape[1] * row[3]))
            y1 = max(0, int(image.shape[0] * row[4]))
            x2 = max(0, int(image.shape[1] * row[5]))
            y2 = max(0, int(image.shape[0] * row[6]))
            outputs.append((x1, y1, x2, y2))
        return outputs