import abc
import os
import cv2

class OpenVINO(abc.ABC):
    def __init__(self, modelDir, modelName):
        self.modelDir = modelDir
        self.modelName = modelName
        self.net = self.load_model()

    def load_model(self):
        model_xml = os.path.join(self.modelDir + self.modelName + ".xml")
        model_bin = os.path.join(self.modelDir + self.modelName + ".bin")
        net = cv2.dnn.readNetFromModelOptimizer(model_xml, model_bin)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    @abc.abstractmethod
    def predict(self, inputs):
        raise NotImplementedError