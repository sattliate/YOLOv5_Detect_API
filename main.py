from Detector.Dector import Detector

import cv2
import os, sys, importlib, time

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
sys.path.append(BASE_PATH)
importlib.reload(sys)


def det_single_image(weights_path, tresh, src_img, dst_img, cls, colors = None):
    det = Detector(weights_path, colors=colors)
    for i in range(2):   #  显卡好，可以多检测几次，保证推理时间准确
        t1 = time.time()

        img = cv2.imread(src_img)
        img_res, det_res = det.detect(img, cls, tresh)

        t2 = (time.time() - t1) * 1000
        print("Using time:{:.3f} ms".format(t2))

        img_res = det.draw_box(img, det_res)
        det.print_result(det_res)
    cv2.imwrite(dst_img, img_res)


if __name__ == '__main__':
    weights_path = "weights/yolov5s.pt"  # 加载你训练好的权重文件
    thresh = 0.45   # 置信度
    src_img = "data/origin/dog.jpg"  # 源照片地址
    dst_img = "data/output/dog_out.jpg"  # 检测结果照片地址
    cls = ['person', 'bus', 'dog', 'bicycle', 'car']  # 检测类别
    colors = {0: (0, 0, 255), 3: (0, 255, 0)}  # box 框的颜色

    det_single_image(weights_path, thresh, src_img, dst_img, cls, colors)
