import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
from pydantic import BaseModel, Extra, Field, validator
from pydantic.fields import PrivateAttr

from frigate.util import load_labels


logger = logging.getLogger(__name__)


class PixelFormatEnum(str, Enum):
    rgb = "rgb"
    bgr = "bgr"
    yuv = "yuv"


class InputTensorEnum(str, Enum):
    nchw = "nchw"
    nhwc = "nhwc"


class ModelTypeEnum(str, Enum):
    ssd = "ssd"
    yolox = "yolox"
    yolov5 = "yolov5"
    yolov8 = "yolov8"
    yolonas='yolonas'


class ModelConfig(BaseModel):
    path: Optional[str] = Field(title="Custom Object detection model path.")
    labelmap_path: Optional[str] = Field(title="Label map for custom object detector.")
    width: int = Field(default=640, title="Object detection model input width.")
    height: int = Field(default=640, title="Object detection model input height.")
    windowsize: int = Field(default=5, title="Window size of motion measure.")
    step: int = Field(default=2, title="Step frame of motion measure.")
    preset_similarity: float = Field(default=0.8, title="Preset similarity threshold.")
    curbbox_motion_threshold: float = Field(default=0.05, title="Motion detected cur bbox obj.")
    bboxes_motion_threshold: float = Field(default=0.05, title="Motion detected bboxes obj.")
    bbox_motion_resize_w: int = Field(default=75, title="width resize bboxes.")
    bbox_motion_resize_h: int = Field(default=75, title="height resize bboxes.")
    path_seg: Optional[str] = Field(title="Custom Object detection model path.")
    segmentor_path: Optional[str] = Field(title="Segmetation model path.")
    labelmap: Dict[int, str] = Field(
        default_factory=dict, title="Labelmap customization."
    )
    input_tensor: InputTensorEnum = Field(
        default=InputTensorEnum.nhwc, title="Model Input Tensor Shape"
    )
    input_pixel_format: PixelFormatEnum = Field(
        default=PixelFormatEnum.rgb, title="Model Input Pixel Color Format"
    )
    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.ssd, title="Object Detection Model Type"
    )
    _merged_labelmap: Optional[Dict[int, str]] = PrivateAttr()
    _colormap: Dict[int, Tuple[int, int, int]] = PrivateAttr()

    @property
    def merged_labelmap(self) -> Dict[int, str]:
        return self._merged_labelmap

    @property
    def colormap(self) -> Dict[int, Tuple[int, int, int]]:
        return self._colormap

    def __init__(self, **config):
        super().__init__(**config)

        self._merged_labelmap = {
            **load_labels(config.get("labelmap_path", "/labelmap.txt")),
            **config.get("labelmap", {}),
        }
        self._colormap = {}

    def create_colormap(self, enabled_labels: set[str]) -> None:
        """Get a list of colors for enabled labels."""
        cmap = plt.cm.get_cmap("tab10", len(enabled_labels))

        for key, val in enumerate(enabled_labels):
            self._colormap[val] = tuple(int(round(255 * c)) for c in cmap(key)[:3])

    class Config:
        extra = Extra.forbid


class BaseDetectorConfig(BaseModel):
    # the type field must be defined in all subclasses
    type: str = Field(default="cpu", title="Detector Type")
    model: ModelConfig = Field(
        default=None, title="Detector specific model configuration."
    )

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True
