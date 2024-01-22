import logging
import numpy as np

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from typing import Literal
from pydantic import Extra, Field

logger = logging.getLogger(__name__)
# import tensorflow
import onnxruntime as ort
import numpy as np
# x, y = test_data[0][0], test_data[0][1]

# outputs = ort_sess.run(None, {'input': x.numpy()})

# # Print Result 
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')
DETECTOR_KEY = "onnx"
class TensorRTDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    # num_threads: int = Field(default=3, title="Number of detection threads")
    device: int = Field(default=0, title="GPU Device Index")

class TensorRTDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: TensorRTDetectorConfig):
        # logger.info(f'{ort.get_available_providers()}')
        self.ort_sess = ort.InferenceSession(detector_config.model.path,providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
        # logger.info(f'{self.ort_sess.get_providers()}')
    def detect_raw(self, tensor_input):
        input = np.array(tensor_input,dtype=np.float32)/255
        logger.info(f'{input.shape}')
        input_name = self.ort_sess.get_inputs()[0].name
        outputs = self.ort_sess.run(None, {input_name:input})
        logger.info(f'{ outputs[0].shape}')
            # Print Result 
        boxes, scores, class_ids = outputs[:][:][0:3], outputs[:][:][4], outputs[:][:][5]
        count = outputs.shape[1]
        detections = np.zeros((20, 8), np.float32)
        logger.info(f'ouput shape: {outputs.shapes}')
        for i in range(count):
            if scores[i] < 0.4 or i == 20:
                break
            detections[i] = [
                class_ids[0][i],
                float(scores[0][i]),
                boxes[0][i][0],
                boxes[0][i][1],
                boxes[0][i][2],
                boxes[0][i][3],
                0.0,
                0.0
            ]

        return detections
