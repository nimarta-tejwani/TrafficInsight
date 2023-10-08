# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import datetime
import torch
import argparse
import pandas as pd
import datetime
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
# from ocr import extract
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
line = [(100, 500), (1050, 500)]
speed_line_queue = {}

def estimatespeed(Location1, Location2):
    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    #distance = speed/time
    speed = d_meters * time_constant
    return int(speed)

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """Simple function that adds fixed color depending on the class"""
    if label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def get_direction(start_point, end_point):
    """
    Function to determine the direction based on the start and end points of a line.
    Assumes start_point and end_point are tuples of (x, y) coordinates.
    Returns the direction as a string.
    """
    x_diff = end_point[0] - start_point[0]
    y_diff = end_point[1] - start_point[1]

    if abs(x_diff) > abs(y_diff):
        if x_diff > 0:
            return "East"
        else:
            return "West"
    else:
        if y_diff > 0:
            return "South"
        else:
            return "North"


def select_lines(img):
    """
    Function to allow the user to select four lines for the directions north, south, east, and west by clicking on the image
    """
    lines = []
    direction_names = ['north', 'south', 'east', 'west']
    current_line = []

    # Create window to display the image
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", img)

    # Mouse callback function to get the starting and ending points of each line
    def select_line(event, x, y, flags, param):
        nonlocal lines, current_line  # Declare 'lines' and 'current_line' as nonlocal to access the outer variables
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(lines) < 4:
                if len(current_line) == 0:
                    cv2.putText(img, direction_names[len(lines)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    current_line.append((x, y))  # Add the starting point of the line
                else:
                    cv2.line(img, (current_line[-1][0], current_line[-1][1]), (x, y), (0, 0, 255), 2)  # Connect the previous point with the current point
                    cv2.imshow("Select ROI", img)
                    current_line.append((x, y))  # Add the ending point of the line

                    if len(current_line) == 2:
                        lines.append((direction_names[len(lines)], current_line))  # Add the completed line to the lines list
                        print(lines)
                        print("current line", current_line)
                        current_line = []  # Reset the current line


    # Set mouse callback function
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_line)

    # Wait for the user to select four lines
    while len(lines) < 4:
        cv2.imshow("Select ROI", img)
        key = cv2.waitKey(1)

        # Check if the user presses the 'q' key to quit the selection process
        if key == ord('q'):
            break

    # Close the window
    cv2.destroyAllWindows()

    return lines


def find_center_point(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


vehicle_entered = []
vehicle_entries = {}

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
   
    if draw_boxes.frame_count == 0:  # Call select_lines method only for the first frame
        lines = select_lines(img)
        draw_boxes.lines = lines
        draw_boxes.frame_count += 1
    else:
        lines = draw_boxes.lines  # Use the lines from the previous call
    print("lines length", len(lines))
    print("out")
    # Draw lines
    for direction, line in lines:
        print("line points", (line[0][0], line[0][1]), (line[1][0], line[1][1]))
        cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (6, 2, 255), 3)
        
        cx, cy = find_center_point(line)
        print(cx, cy)
        cv2.putText(img, direction, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

    # extracted_date = extract(img)
    
    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          print("id", id)
          data_deque[id] = deque(maxlen= 64)
          speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        print('obj_name', obj_name)
        
        # if obj_name == 'car':
        #     obj_name = 'car/taxi'
        # elif obj_name == 'bicycle':
        #     obj_name = 'light vehicle'
        # elif obj_name == 'motorcycle':
        #     obj_name = 'light vehicle'
        # elif obj_name=='truck':
        #     obj_name='truck'
        # elif obj_name=='bus':
        #     obj_name='bus'

        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        print(data_deque[id])
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)

            for d, line in lines:
                if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                    if id in vehicle_entries:
                        v = vehicle_entries[id]
                        v['Exit Point'] = d
                    else:
                        v = {
                            'ID': id,
                            'Type': obj_name,
                            'Entry Point': d,
                            'Exit Point': None,
                            'Entry Time': datetime.datetime.now()
                        }
                        vehicle_entries[id] = v

        print("Vehicle Entry Data:")
        for vehicle_id, vehicle_entry in vehicle_entries.items():
            print(vehicle_entry)

        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
        
    print("vehicles entries", vehicle_entries)
    
    df = pd.DataFrame.from_dict(vehicle_entries, orient='index')
    filename = "output.csv"
    df.to_csv(filename, index=False)

    return img


# Initialize frame_count and lines as attributes of the draw_boxes function
draw_boxes.frame_count = 0
draw_boxes.lines = []

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
      
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
      
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            print("log string", log_string)
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = "my_video.mp4"
    predictor = DetectionPredictor(cfg)
    predictor()
