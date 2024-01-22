#!/usr/bin/env python3
import os
import time
import logging
import argparse
import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
from cuda import cudart
import ctypes
from typing import Optional, List, Tuple
# from skimage import io
import cv2
# from skimage.transform import resize
import logging
# import torchvision
# import torch
import ctypes
import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda

    TRT_SUPPORT = True
except ModuleNotFoundError as e:
    TRT_SUPPORT = False

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from typing import Literal
from pydantic import Field

logger = logging.getLogger(__name__)

DETECTOR_KEY = "segmentor"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


def cuda_call(call):
    err, res = call[0], call[1:]
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype, shape: Tuple[int, int, int]):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        self.shape = shape
        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
    @property
    def host(self) -> np.ndarray:
        return self._host
    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')
    @property
    def device(self) -> int:
        return self._device
    @property
    def nbytes(self) -> int:
        return self._nbytes
    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"
    def __repr__(self):
        return self.__str__()
    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

class TensorRTSegmentorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: int = Field(default=0, title="GPU Device Index")


class TensorRtSegmentor(DetectionApi):
    type_key = DETECTOR_KEY

    trt_logger = trt.Logger()

    def __init__(self, detector_config: TensorRTSegmentorConfig):
        trt.init_libnvinfer_plugins(None, "")
        trt_path = os.path.splitext(detector_config.model.segmentor_path)[0] + ".trt"
        # print(trt_path)
        if (not os.path.exists(trt_path)):
            print("building engine from ", detector_config.model.segmentor_path)
            self.engine = self.build_engine_from_onnx(detector_config.model.segmentor_path,fp16=True)
            with open(trt_path, "wb") as fw:
                fw.write(self.engine.serialize())
            print("saved engine to ", trt_path)
        else:
            print("reading ", trt_path)
            with open(trt_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)


    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    def allocate_buffers(self, engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [str(i) for i in self.engine]
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = engine.get_binding_shape(binding) if profile_idx is None else engine.get_binding_profile_shape(binding, profile_idx)[-1]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
            if not engine.binding_is_input(binding):
                dtype = np.dtype(np.float32)
            # Allocate host and device buffers
            bindingMemory = HostDeviceMem(size, dtype, shape)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))

            # Append to the appropriate list.
            
            if engine.binding_is_input(binding):
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream


    def build_engine_from_onnx(self, onnx_file_path, fp16=False):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.trt_logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             builder.create_builder_config() as config, \
             trt.OnnxParser(network, self.trt_logger) as parser, \
             trt.Runtime(self.trt_logger) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2Gb
            if fp16:
                print("setting fp16 flag")
                config.set_flag(trt.BuilderFlag.FP16)
            # Parse model file
            assert os.path.exists(onnx_file_path), f'cannot find {onnx_file_path}'
            with open(onnx_file_path, 'rb') as fr:
                if not parser.parse(fr.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    assert False
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            return engine


    def infer(self, image):
        self.inputs[0].host = image
        # print(self.inputs[0].host)
        # Transfer input data to the GPU.
        host2device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, host2device, self.stream)) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        # Transfer predictions back from the GPU.
        device2host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, device2host, self.stream)) for out in self.outputs]
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        # Return outputs
        return [out.host for out in self.outputs]

    def detect_raw(self, tensor_input):
        image_array = np.array(tensor_input/255).astype(np.float32)
        input = self._scale_image(image_array = image_array,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # input.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        trt_outputs = self.infer(image=input)
        # print(trt_outputs)
        mask = trt_outputs[0]
        

        mask = np.reshape(mask,(640,640)).round()
        # mask_img = mask*255
        # import cv2
        # cv2.imwrite('/media/frigate/mask.png',mask_img)
        return mask
    
    def _scale_image(self,image_array, mean, std):
        """
        Scale an RGB image represented as a NumPy array with mean and standard deviation.

        Args:
            image_array (numpy.ndarray): Input image represented as a NumPy array with shape (height, width, channels).
            mean (tuple): Mean values for each channel (R, G, B).
            std (tuple): Standard deviation values for each channel (R, G, B).

        Returns:
            numpy.ndarray: Scaled image represented as a NumPy array.
        """
        # Convert the image array to float32 (if not already)
        image_array = image_array.astype(np.float32)

        # Scale the image
        scaled_image = (image_array - np.array(mean).reshape(1, -1, 1, 1)) / np.array(std).reshape(1, -1, 1, 1)

        return scaled_image.astype(np.float32)