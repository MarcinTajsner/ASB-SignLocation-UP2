import time
import torch
import torchvision
import cv2
import yaml
import numpy as np
from openvino.runtime import Core

class InferenceEngine:

    def __init__(
        self,
        model_path: str,
        weights_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detail: int = 300,
        device: str = "HDDL",
        img_resolution: int = 1280,
        img_stride: int = 64,
        img_border_color: tuple = (114, 114, 114)
    ):
        #Assign class variables
        self.__conf_thres = confidence_threshold
        self.__iou_thres = iou_threshold
        self.__max_det = max_detail
        self.__img_shape = img_resolution
        self.__img_stride = img_stride
        self.__img_border_color = img_border_color

        #Model initialization
        self.__ie = Core()
        self.__model = self.__ie.read_model(model=model_path, weights=weights_path)
        self.__model = self.__ie.compile_model(model=self.__model, device_name=device)
        self.__model_output_layer = next(iter(self.__model.outputs))

    def __clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def __scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.__clip_coords(coords, img0_shape)
        return coords
            
    def __xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def __xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def __non_max_suppression(self, prediction, agnostic=False, multi_label=False, labels=()):
        # Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
        # Returns: list of detections, on (n,6) tensor per image [xyxy, conf, cls]

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.__conf_thres  # candidates

        # Checks
        assert 0 <= self.__conf_thres <= 1, f'Invalid Confidence threshold {self.__conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.__iou_thres <= 1, f'Invalid IoU {self.__iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.3 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.__xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self.__conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.__conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.__iou_thres)  # NMS
            if i.shape[0] > self.__max_det:  # limit detections
                i = i[:self.__max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > self.__iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    def __img_preprocessing(self, orginal_img, auto=True, scaleFill=False, scaleup=True):
        # Resize and pad image while meeting stride-multiple constraints
        img = orginal_img.copy()
        shape = orginal_img.shape[:2]  # current shape [height, width]
        new_shape = (self.__img_shape, self.__img_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup: 
            r = min(r, 1.0)  # only scale down, do not scale up (for better val mAP)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.__img_stride), np.mod(dh, self.__img_stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad: 
            img = cv2.resize(orginal_img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.__img_border_color)  # add border

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cpu")
        img = img.half()  # to float16
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3: img = img[None]  # expand for batch dim

        return img

    def __predict(self, img):
        pred = self.__model([img])[self.__model_output_layer]
        pred = torch.tensor(pred, device="cpu")
        return pred

    def __prediction_postprocessing(self, pred, img, orginal_img):
        pred = self.__non_max_suppression(pred)
        pred = pred[0]
        pred[:, :4] = self.__scale_coords(img.shape[2:], pred[:, :4], orginal_img.shape).round()
        pred_list = []
        for p in pred:
            pred_list.append({
                "xmin": float(p[0]),
                "ymin": float(p[1]),
                "xmax": float(p[2]),
                "ymax": float(p[3]),
                "conf": float(p[4]),
                "class": int(p[5])
            })
        return pred_list

    def detect_from_frame(self, frame) -> list:
        img = self.__img_preprocessing(frame)
        pred = self.__predict(img)
        pred_list = self.__prediction_postprocessing(pred, img, frame)
        return pred_list


    












