import math
import torch
import cv2
from mss import mss
from PIL import Image
import torch
import numpy as np
import ffd
import pyautogui
# Screen capture
sct = mss()
SCREEN_W = 1920  # 屏幕长
SCREEN_H = 1080  # 屏幕高
SCREEN_CX = SCREEN_W // 2  # 屏幕中心x
SCREEN_CY = SCREEN_H // 2  # 屏幕中心y
SCREEN_C = [SCREEN_CX, SCREEN_CY]  # 屏幕中心坐标
SCREENSHOT_W = 800  # 截图区域长
SCREENSHOT_H = 640  # 截图区域高
LEFT = SCREEN_CX - SCREENSHOT_W // 2  # 检测框左上角x
TOP = SCREEN_CY - SCREENSHOT_H // 2  # 检测框左上角y

def Center(p):
    """
    返回中心坐标;
    :param p: [lx,ly,w,h]->[左上x坐标，左上y坐标]
    :return: [x,y]
    """
    return [p[0] + p[2] // 2, p[1] + p[3] // 2]


def Distence(a, b):
    """
    两点间距离
    :param a:a点 (xa,ya)
    :param b: b点(xb,yb)
    :return: sqrt((xa-xb)**2 + (yb-ya)**2)
    """
    return math.sqrt(
        ((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

def FindBestCenter(detections):
    """
    根据检测的结果，寻找最佳射击坐标
    :param detections: 检测结果
    :return: 最佳射击坐标
    """
    ch = {'p': [0, 0, 0, 0], 'd': float('inf'), 'c': 0.0}  # 离枪口最近的头 p位置 d距离中心距离 c可信度
    cp = {'p': [0, 0, 0, 0], 'd': float('inf'), 'c': 0.0}  # 最枪口近的身子 p位置 d距离中心距离 c可信度
    for dt in detections:
        # {'class': 'helmet', 'conf': 0.6904296875, 'position': [266, 468, 50, 85]}
        if dt['conf'] > 0.30:  # 只寻找置信度达到70%的头和身子
            dt_p = dt['position']  # 检测出来的目标位置
            dt_c = Center(dt_p)  # w,h [291, 510]
            # print(dt['class'])
            if dt['class'] == 'helmet':  # 判断是不是最优头
                # 669.6723079238084
                dt_d = Distence(dt_c, SCREEN_C)
                # {'p': [540, 440, 86, 121], 'd': 379.1160772111887, 'c': 0.85009765625}
                if dt_d < ch['d']:
                    ch['p'] = dt['position']
                    ch['d'] = dt_d
                    ch['c'] = dt['conf']
                    pass

            # if dt['class'] == 'person':  # 判断是不是最优身子
                # dt_d = Distence(dt_c, SCREEN_C)
                # if dt_d < cp['d']:
                #     cp['p'] = dt['position']
                #     cp['d'] = dt_d
                #     cp['c'] = dt['conf']
                #     pass

    if cp['d'] < float('inf') or ch['d'] < float('inf'):  # 自动选择瞄准部位
        btp = ch['p'] if ch['c'] > cp['c'] else cp['p']  # best target position
        btc = Center(btp)  # best target center
        return btc, btp
    return None, None

def tttee():

    ckpt_path=r"E:\pythonProject\yolov5\runs\train\exp\weights\best.pt"
    model = torch.hub.load("E:\pythonProject\yolov5",
                           "custom",
                           path=ckpt_path,
                           source='local'
                           )


    while 1:
        # set the capture size
        w, h = 800, 640
        # set the capture position
        monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
        img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
        screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # set the model use the screen
        result = model(screen)
        result.display(render=True)

        result.show()
        FindBestCenter(result)
        # show the result
        # cv2.imshow('Screen', result.imgs[0])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    while 1:
        # set the capture size
        # w, h = 800, 640
        # set the capture position
        monitor = {'top': 0, 'left': 0, 'width': SCREENSHOT_W, 'height': SCREENSHOT_W}
        img = Image.frombytes('RGB', (SCREENSHOT_W, SCREENSHOT_W), sct.grab(monitor).rgb)
        screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # set the model use the screen
        # 送入yolo检测
        detections = ffd.detect(screen)
        # 确定目标最优的射击中心
        btc, btp = FindBestCenter(detections)

        # [540, 440, 86, 121]
        # [583, 500]
        # 如果屏幕区域有射击目标
        if btc is not None:
            # dll.MoveTo2(int(LEFT + btc[0]), int(TOP + btc[1]))  # 调用易键鼠移动鼠标（此处更换为自己的）
            pyautogui.moveTo(int( btc[0]), int(btc[1]))


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break