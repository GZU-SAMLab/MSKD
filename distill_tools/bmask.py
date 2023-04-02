import torch


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bmask(images, targets):
    device = images.device

    _, _, h, w = images.shape  # batch size, _, height, width
    bm = torch.zeros(images.shape).to(device)
    # 处理每张图片
    for i, img in enumerate(images):

        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i] # targets.shap(在哪张图，类， x, y, w, h)
            boxes = xywh2xyxy(image_targets[:, 2:6]).T # 标签元素顺序 x y w h

            boxes[[0, 2]] *= w  # x1 x2
            boxes[[1, 3]] *= h  # y1 y2
            
            x1, x2 = boxes[[0, 2]]
            y1, y2 = boxes[[1, 3]]
            # 处理每个真实框
            for bb in range(boxes.shape[1]):
                bm[i, :,int(y1[bb]):int(y2[bb]), int(x1[bb]):int(x2[bb])] = 1
    return bm