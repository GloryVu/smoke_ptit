import datetime
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
from abc import ABC, abstractmethod
import cv2
import numpy as np
from setproctitle import setproctitle

from frigate.config import InputTensorEnum
from frigate.detectors import create_detector, create_segmentor

from frigate.util import EventsPerSecond, SharedMemoryFrameManager, listen, load_labels, intersection_over_union, draw_box_with_label

logger = logging.getLogger(__name__)


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, tensor_input, threshold=0.4):
        pass


def tensor_transform(desired_shape):
    # Currently this function only supports BHWC permutations
    if desired_shape == InputTensorEnum.nhwc:
        return None
    elif desired_shape == InputTensorEnum.nchw:
        return (0, 3, 1, 2)


class LocalObjectDetector(ObjectDetector):
    def __init__(
        self,
        detector_config=None,
        labels=None,
    ):
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels = {}
        else:
            self.labels = load_labels(labels)

        if detector_config:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)
        else:
            self.input_transform = None

        self.detect_api = create_detector(detector_config)

    def detect(self, tensor_input, threshold=0.4):
        detections = []

        raw_detections = self.detect_raw(tensor_input)

        for d in raw_detections:
            if d[1] < threshold:
                break
            detections.append(
                (self.labels[int(d[0])], float(d[1]), (d[2], d[3], d[4], d[5]),(d[6],d[7]))
            )
        self.fps.update()
        return detections

    def detect_raw(self, tensor_input):
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)
        return self.detect_api.detect_raw(tensor_input=tensor_input)

class LocalSegmentor(ObjectDetector):
    def __init__(
        self,
        detector_config=None,
        labels=None,
    ):
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels = {}
        else:
            self.labels = load_labels(labels)

        if detector_config:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)
        else:
            self.input_transform = None

        self.detect_api = create_segmentor(detector_config)

    def detect(self, tensor_input, threshold=0.4):
        return self.detect_raw(tensor_input)
        
    def detect_raw(self, tensor_input):
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)
        return self.detect_api.detect_raw(tensor_input=tensor_input)

def run_detector(
    name: str,
    detection_queue: mp.Queue,
    out_events: dict[str, mp.Event],
    avg_speed,
    start,
    detector_config,
):
    def similarity(img1,img2):
        # logger.info(f'{img1} {img2}')
                    
        # logger.info(img1.shape)
        # img1 = np.squeeze(img1)
        # img2 = np.squeeze(img2)
        # logger.info(f'{img1.shape}')
        # logger.info(f'{img2}\n---------------------')
        
        if(img1.shape!=img2.shape):
            img1 = cv2.resize(img1,[img2.shape[1],img2.shape[0]])
        # logger.info(f' image {img1}{img2}')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        # logger.info(f'{gray1}')
        diff = cv2.absdiff(gray1,gray2)
        # logger.info(f'difference error: {absdiff}')
        similarity = 1 - diff.sum() / float(img1.shape[0]*img1.shape[1]*255)
        return similarity
    
    def motion_measure(input_motion):
        sim = 0
        input_motion = [cv2.resize(im,(detector_config.model.bbox_motion_resize_h,detector_config.model.bbox_motion_resize_w))
                        for im in input_motion]
        for i in range(len(input_motion)-1):
            # logger.info(f'diff mmb {similarity(input_motion[i],input_motion[i+1])}')
            sim += similarity(input_motion[i],input_motion[i+1])
        # logger.log(input_motion)
        return 1-sim/(len(input_motion)-1)
    
    def get_bbox(detection,h,w):
        return [max(0,int(detection[2]*w//1)),
                max(0,int(detection[3]*h//1)),
                min(w-1,int(detection[4]*w//1)),
                min(h-1,int(detection[5]*h//1))]
    
    def find_matchest_box(box,predet,h,w):
        max_iou = 0
        matchest_box = []
        for det in predet:
            if det[1] == 0:
                continue
            if(label_map[detection[0]]!='smoke'):
                continue
            prebox = get_bbox(det,h,w)
            iou = intersection_over_union(box,prebox)
            if (iou > max_iou):
                max_iou = iou
                matchest_box = prebox
        return max_iou,matchest_box
    threading.current_thread().name = f"detector:{name}"
    logger = logging.getLogger(f"detector.{name}")
    logger.info(f"Starting detection process: {os.getpid()}")
    setproctitle(f"frigate.detector.{name}")
    listen()
    
    stop_event = mp.Event()
    def on_sky(box,mask,threshold=0.8):
        percent = 1.0*mask[box[1]:box[3]][box[0]:box[2]].sum()/((box[3]-box[1])*(box[2]-box[0]))
        return percent >= threshold
    def receiveSignal(signalNumber, frame):
        logger.info("Signal to exit detection process...")
        stop_event.set()

    signal.signal(signal.SIGTERM, receiveSignal)
    signal.signal(signal.SIGINT, receiveSignal)

    frame_manager = SharedMemoryFrameManager()
    object_detector = LocalObjectDetector(detector_config=detector_config)
    segmentor = LocalSegmentor(detector_config=detector_config)
    # logger.info(f'{detector_config}')
    wd_size = detector_config.model.windowsize
    step = detector_config.model.step
    outputs = {}
    frame_queue ={}
    output_queue={}
    segmentors={}
    for name in out_events.keys():
        out_shm = mp.shared_memory.SharedMemory(name=f"out-{name}", create=False)
        out_np = np.ndarray((20, 8), dtype=np.float32, buffer=out_shm.buf)
        out_q_shm = mp.shared_memory.SharedMemory(name=f"out_q-{name}", create=False)
        out_q_np = np.ndarray((wd_size,20, 8), dtype=np.float32, buffer=out_q_shm.buf)
        frame_q_shm = mp.shared_memory.SharedMemory(name=f"frame_q-{name}", create=False)
        frame_q_np = np.ndarray((wd_size,detector_config.model.height, detector_config.model.width,3), dtype=np.uint8, buffer=frame_q_shm.buf)
        segmentor_shm = mp.shared_memory.SharedMemory(name=f"segmentor-{name}", create=False)
        segmentor_np = np.ndarray((detector_config.model.height, detector_config.model.width,1), dtype=np.uint8, buffer=segmentor_shm.buf)
        
        outputs[name] = {"shm": out_shm, "np": out_np}
        output_queue[name] = {'shm':out_q_shm,'np':out_q_np}
        frame_queue[name] = {'shm':frame_q_shm,'np':frame_q_np}
        segmentors[name] = {'shm':segmentor_shm,'np':segmentor_np}
        # logger.info(f'{name}')
        

    # frame_queue =[]
    # output_queue =[]
    label_map = detector_config.model.labelmap

    c = 0
    sky_object_count = 0
    static_object_count = 0
    hard_neg_samp_count = 0
    while not stop_event.is_set():
        try:
            connection_id = detection_queue.get(timeout=1)
        
            # logger.info(f'connection_id {connection_id}')
        except queue.Empty:
            continue
        input_frame = frame_manager.get(
            connection_id,
            (1, detector_config.model.height, detector_config.model.width, 3),
        )

        if input_frame is None:
            continue
        ouput_frame = np.squeeze(input_frame)
        # cv2.imwrite(f'/media/frigate/clips/detect{c}-{connection_id}.jpg',ouput_frame)
        c+=1
        start.value = datetime.datetime.now().timestamp()
        # logger.info(f'{input_frame}')
        detections = object_detector.detect_raw(input_frame)
        
        # cv2.imwrite(f'/media/frigate/clips/detect{count}.jpg',np.squeeze(input_frame))
        duration = datetime.datetime.now().timestamp() - start.value
        # logger.info(msg = detections[0])
        #     detections[i] = [
        #     class_ids[i],
        #     float(scores[i]),
        #     boxes[i][0],
        #     boxes[i][1],
        #     boxes[i][2],
        #     boxes[i][3],
        # ]
        # detect and send the output
        # detect and send the output
        for i in range(wd_size-1):
            frame_queue[connection_id]['np'][i][:] = frame_queue[connection_id]['np'][i+1][:]
            output_queue[connection_id]['np'][i][:] = output_queue[connection_id]['np'][i+1][:]
        frame_queue[connection_id]['np'][wd_size-1][:] = np.squeeze(input_frame.copy())[:]
        output_queue[connection_id]['np'][wd_size-1][:] =detections.copy()[:]
        # logger.info(f'queue len: {len(output_queue)}' )
        # logger.info(f'frame queue: {frame_queue}' )
        
        if(frame_queue[connection_id]['np'][0].sum()!=0):
            # dev_vinh: detect
            # check exist detections
            # logger.info(f'{frame_queue[connection_id]}')
            preset_similarity = similarity(frame_queue[connection_id]['np'][0],frame_queue[connection_id]['np'][wd_size-1])
            # logger.info(f'preset similarity camera {connection_id} : {preset_similarity}')
            if(preset_similarity>=detector_config.model.preset_similarity):
                # remove no motion objects
                labels = ''
                for i, detection in enumerate(detections):
                    if detection[1] == 0:
                        continue
                    if(label_map[int(detection[0])]!='smoke'):
                        detections[i] = [0.0]*8
                        continue
                    h = input_frame.shape[1]
                    w = input_frame.shape[2]
                    
                    box = get_bbox(detection,h,w)
                    if detection[1] > 0.5:
                        labels+= f'1 {1.0*(box[0]+box[2])/2/640} {1.0*(box[1]+box[3])/2/640} {1.0*((box[2]-box[0])/640)} {1.0*((box[3]-box[1])/640)}\n'
                    if on_sky(box, segmentors[connection_id]['np']):
                        
                        draw_box_with_label(ouput_frame,box[0],box[1],box[2],box[3],'on_sky_object',f'{int(detection[1]*100)}%')
                        # cv2.imwrite(f'/media/frigate/on_sky_object/detect{sky_object_count}-{connection_id}.jpg',ouput_frame)
                        sky_object_count+=1
                        detections[i] = [0.0]*8
                        continue
                    # logger.info(f'boxxxxxxxxx {box}')
                    print(frame_queue[connection_id]['np'].shape)
                    input_motion = [a[box[1]:box[3],box[0]:box[2],:].copy() for a in frame_queue[connection_id]['np'][::step]]
                    # logger.info(f'{input_motion}')
                    # logger.info(f'inputmotion shape : {input_motion[0].shape}')
                    # logger.info(f'box : {box}')
                    # for ii in range(len(input_motion)):
                    #     input_motion[ii] = input_motion[ii][box[1]:box[3],box[0]:box[2],:].copy()
                    # logger.info(f'inputmotion strimed shape : {input_motion[0].shape}')
                    mm = motion_measure(input_motion)
                    # logger.info(f'motion_measure : {mm} {detector_config.model.curbbox_motion_threshold}')
                    if mm < detector_config.model.curbbox_motion_threshold:
                        
                        draw_box_with_label(ouput_frame,box[0],box[1],box[2],box[3],'static',f'{int(detection[1]*100)}% {int(mm*100)}%')
                        # cv2.imwrite(f'/media/frigate/static_motionless_object/detect{static_object_count}-{connection_id}.jpg',ouput_frame)
                        static_object_count+=1
                        detections[i] = [0.0]*8
                        continue
                    # check previous detection
                    input_motion_box = [frame_queue[connection_id]['np'][-1][box[1]:box[3],box[0]:box[2],:].copy()]
                    for j in range(0,wd_size-1,step):
                        predet = output_queue[connection_id]['np'][wd_size-2-j]
                        iou, matchest_box = find_matchest_box(box,predet,h,w)
                        # logger.info(f'{iou} {matchest_box}')
                        if(iou < 0.3):
                            break
                        # current_frame = 
                        input_motion_box.append(
                        frame_queue[connection_id]['np'][wd_size-2-j][matchest_box[1]:matchest_box[3],matchest_box[0]:matchest_box[2],:].copy())
                   
                    if(len(input_motion_box)<= wd_size//(2*step)):
                        detections[i] = [0.0]*8
                        continue
                    # logger.info(f'inputmotionbox shape : {input_motion_box}')
                    # logger.info(f'inputmotionbox shape : {len(input_motion_box)}')
                    mmb = motion_measure(input_motion_box)
                    # logger.info(f'motion_measure_box : {mmb} {detector_config.model.bboxes_motion_threshold}')
                    
                    if(mmb<detector_config.model.bboxes_motion_threshold):
                        draw_box_with_label(ouput_frame,box[0],box[1],box[2],box[3],'motionless',f'{int(detection[1]*100)}% {int(mm*100)}%')
                        # cv2.imwrite(f'/media/frigate/static_motionless_object/detect{static_object_count}-{connection_id}.jpg',ouput_frame)
                        static_object_count+=1
                        detections[i] = [0.0]*8
                        continue
                    
                    draw_box_with_label(ouput_frame,box[0],box[1],box[2],box[3],'smoke',f'{int(detection[1]*100)}% {int(mm*100)}% {int(mmb*100)}#')
                    # cv2.imwrite(f'/media/frigate/clips/detect{c}-{connection_id}.jpg',ouput_frame)
                    c+=1
                    detections[i][6] = mm
                    detections[i][7] = mmb
                # if labels != '':
                #     img = cv2.cvtColor(ouput_frame, cv2.COLOR_BGR2RGB)
                #     cv2.imwrite(f'/media/frigate/hard_negative_samples/not_smoke_{hard_neg_samp_count}-{connection_id}.jpg',img)
                #     with open(f'/media/frigate/hard_negative_samples/not_smoke_{hard_neg_samp_count}-{connection_id}.txt','w') as f:
                #         f.write(labels)
                #     hard_neg_samp_count+=1
                outputs[connection_id]["np"][:] = detections[:]
            else:
                segmentations = segmentor.detect_raw(input_frame)
                mask_img = segmentations*255
                # import cv2
                # cv2.imwrite(f'/media/frigate/mask/{connection_id}-mask.png',mask_img) 
                # cv2.imwrite(f'/media/frigate/mask/{connection_id}-image.png',ouput_frame) 
                segmentors[connection_id]['np'] = segmentations
                outputs[connection_id]["np"][:] = np.zeros((20, 8), np.float32)[:]
            # logger.info(f'{output_queue[1]}')
            # frame_queue.pop(0)
            # output_queue.pop(0)
        else: 
            segmentations = segmentor.detect_raw(input_frame)
            mask_img = segmentations*255
            # import cv2 ouput_frame
            # cv2.imwrite(f'/media/frigate/mask/{connection_id}-mask.png',mask_img) 
            # cv2.imwrite(f'/media/frigate/mask/{connection_id}-image.png',ouput_frame) 
            segmentors[connection_id]['np'] = segmentations
            outputs[connection_id]["np"][:] = np.zeros((20, 8), np.float32)[:]
            #np.zeros((20, 6), np.float32)[:]
        # logger.info(f'{outputs[connection_id]["np"]}')
        out_events[connection_id].set()
        start.value = 0.0
        avg_speed.value = (avg_speed.value * 9 + duration) / 10
        # logger.info('end frame')
    logger.info("Exited detection process...")


class ObjectDetectProcess:
    def __init__(
        self,
        name,
        detection_queue,
        out_events,
        detector_config,
    ):
        self.name = name
        self.out_events = out_events
        self.detection_queue = detection_queue
        self.avg_inference_speed = mp.Value("d", 0.01)
        self.detection_start = mp.Value("d", 0.0)
        self.detect_process = None
        self.detector_config = detector_config
        self.start_or_restart()

    def stop(self):
        # if the process has already exited on its own, just return
        if self.detect_process and self.detect_process.exitcode:
            return
        self.detect_process.terminate()
        logging.info("Waiting for detection process to exit gracefully...")
        self.detect_process.join(timeout=30)
        if self.detect_process.exitcode is None:
            logging.info("Detection process didnt exit. Force killing...")
            self.detect_process.kill()
            self.detect_process.join()
        logging.info("Detection process has exited...")

    def start_or_restart(self):
        self.detection_start.value = 0.0
        if (not self.detect_process is None) and self.detect_process.is_alive():
            self.stop()
        self.detect_process = mp.Process(
            target=run_detector,
            name=f"detector:{self.name}",
            args=(
                self.name,
                self.detection_queue,
                self.out_events,
                self.avg_inference_speed,
                self.detection_start,
                self.detector_config,
            ),
        )
        self.detect_process.daemon = True
        self.detect_process.start()


class RemoteObjectDetector:
    def __init__(self, name, labels, detection_queue, event, model_config, stop_event):
        self.labels = labels
        self.name = name
        self.fps = EventsPerSecond()
        self.detection_queue = detection_queue
        self.event = event
        self.stop_event = stop_event
        self.shm = mp.shared_memory.SharedMemory(name=self.name, create=False)
        self.np_shm = np.ndarray(
            (1, model_config.height, model_config.width, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        self.out_shm = mp.shared_memory.SharedMemory(
            name=f"out-{self.name}", create=False
        )
        self.out_np_shm = np.ndarray((20, 8), dtype=np.float32, buffer=self.out_shm.buf)
        self.out_q_shm = mp.shared_memory.SharedMemory(
            name=f"out_q-{self.name}", create=False
        )
        self.frame_q_shm = mp.shared_memory.SharedMemory(
            name=f"frame_q-{self.name}", create=False
        )
    def detect(self, tensor_input, threshold=0.4):
        detections = []

        if self.stop_event.is_set():
            return detections
        # logger.info(f'{tensor_input.shape}')
        # cv2.imwrite('/media/frigate/clips/detect.jpg',tensor_input)
        # copy input to shared memory
        self.np_shm[:] = tensor_input[:]
        self.event.clear()
        self.detection_queue.put(self.name)
        result = self.event.wait(timeout=5.0)

        # if it timed out
        if result is None:
            return detections

        for d in self.out_np_shm:
            if d[1] < threshold:
                break
            detections.append(
                (self.labels[int(d[0])], float(d[1]), (d[2], d[3], d[4], d[5]),(d[6],d[7]))
            )
        self.fps.update()
        # logger.info(f'return detection {detections}')
        return detections

    def cleanup(self):
        self.shm.unlink()
        self.out_shm.unlink()
        self.frame_q_shm.unlink()
        self.out_q_shm.unlink()