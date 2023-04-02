import numpy as np
import cv2
from utils.general import xywh2xyxy

def gen_binary_scale_mask(shape, label):
    label = xywh2xyxy(label[:,2:])
    w, h, _ = shape
    bmask = np.zeros((w,h), dtype=np.float32)
    smask = np.zeros((w,h), dtype=np.float32)
    s_sum = 0.0
    x1, y1, x2, y2 = label[:,0], label[:,1], label[:,2], label[:,3]
    for i in range(label.shape[0]):
        bmask = cv2.rectangle(bmask, [x1[i], y1[i]], [x2[i], y2[i]], 1, -1)
        s = 1.0 * (x2[i] - x1[i]) * (y2[i] - y1[i])
        s_sum += s
        smask = cv2.rectangle(smask, [x1[i], y1[i]], [x2[i], y2[i]], 1/s, -1)
    smask[smask==0] = 1.0 / s_sum
    return bmask, smask