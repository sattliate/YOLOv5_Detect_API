import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.plots import plot_one_box
import random


class Detector:

    def __init__(self, weights_path, colors=None):
        self.weights = weights_path
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()  #  model.eval() 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
        model.float() #开启半精度。直接可以加快运行速度、减少GPU占用，并且只有不明显的accuracy损失

        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names

        # 颜色框的种类颜色
        # !!! 调试：[random.randint(0, 255) for _ in range(3)]
        # print([random.randint(0, 255) for _ in range(3)],[[random.randint(0, 255) for _ in range(3)] for _ in self.names ])
        self.colors_random = [[random.randint(0, 255) for _ in range(3)] for _ in self.names ]

        # 如果设置了颜色就用设置的，未用随机生成用上面代码的
        self.colors = colors
        if self.colors:
            for color in self.colors:
                self.colors_random[color] = self.colors[color]

    def preprocess(self, img, img_size):  # 预处理

        img0 = img.copy() # 原始图像
        img = letterbox(img,new_shape=img_size)[0] # resize 图片大小
        # print(img[:, :, ::-1], img[:, :, ::-1].transpose(2, 0, 1))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img) # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() ==3:
            img = img.unsqueeze(0) # 处理后的图像
        return img0, img

    def detect(self, im, cls, tresh, img_size=640):

        im0, img = self.preprocess(im, img_size)
        pred = self.m(img,augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, tresh, 0.4) # 非极大抑制
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round() #     Rescale coords (xyxy) from img1_shape to img0_shape

                # # 打印一下det
                # print(det)
                for *x, conf, cls_id in det:
                    label_name = self.names[int(cls_id)]
                    if not label_name in cls: # 检测的cls_id 不在类库中
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, cls_id, label_name, conf))
        return im, pred_boxes


    def draw_box(self, im, pred_boxes):
        if not pred_boxes:
            print("未检测到分类框集合，无法画框  ")
        else:
            for pred_box in pred_boxes:
                plot_one_box(pred_box, im, label=pred_box[5] +':{:.2f} '.format(pred_box[6]),
                             color=self.colors_random[int(pred_box[4])], line_thickness=3)
        return im


    def print_result(self, pred_boxes):
        if not pred_boxes:
            print("未检测到分类框集合，无法打印结果!!")
        else:
            for i in range(len(pred_boxes)):
                print("{}st object is {}, class id is {:.0f}, x:{},y:{},w:{},h:{},conf:{}".format(i, pred_boxes[i][5],
                                                                                          pred_boxes[i][4],
                                                                                          pred_boxes[i][0],
                                                                                          pred_boxes[i][1],
                                                                                          (pred_boxes[i][2] -
                                                                                           pred_boxes[i][0]),
                                                                                          (pred_boxes[i][3] -
                                                                                           pred_boxes[i][1]),
                                                                                          pred_boxes[i][6]))