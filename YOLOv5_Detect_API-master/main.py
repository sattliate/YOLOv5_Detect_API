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
    weights_path = r"D:\Programing\PythonProject\PyTorch\weights\0822P067R06best.pt"
    thresh = 0.45
    src_img = r"data/origin/bus.jpg"
    dst_img = "data/output/bus_out.jpg"
    cls = ['no-mask', 'mask']
    colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # box 框的颜色
    det_single_image(weights_path, thresh, src_img, dst_img, cls, colors)
