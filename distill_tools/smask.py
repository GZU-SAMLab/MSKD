import torch
from distill_tools.bmask import xywh2xyxy


def smask(images, targets):
    device = images.device
    
    _, _, h, w = images.shape  # batch size, _, height, width
    sm = torch.full(images.shape, -1,dtype=torch.float).to(device)
    # 处理每张图片
    for i, _ in enumerate(images):

        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i] # targets.shap(在哪张图，类， x, y, w, h)
            boxes = xywh2xyxy(image_targets[:, 2:6]).T # 标签元素顺序 x y w h

            boxes[[0, 2]] *= w  # x1 x2
            boxes[[1, 3]] *= h  # y1 y2
            x1, x2 = boxes[[0, 2]]
            y1, y2 = boxes[[1, 3]]
            
            # 处理每个真实框 scale mask 前景
            area_array = {}
            for bb in range(boxes.shape[1]):
                area_array[bb] = 1 / float((y2[bb] - y1[bb]) * (x2[bb] - x1[bb]))

            for key, value in sorted(area_array.items(), key=lambda item:item[1]):
                sm[i, :,int(y1[key]):int(y2[key]), int(x1[key]):int(x2[key])] = value
            
            # 处理每个真实框 scale mask 背景
            sm[sm == -1] = 1 / (sm == -1).sum()  
    return sm