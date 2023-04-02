import torch
from torch.nn.functional import softmax


# 获取通道注意力掩码
def channel_mask(featurs, T =1):
    # featurs.shape N C H W
    N, C, H, W = featurs.shape
    cgraph = torch.abs(featurs).sum(dim=[2, 3]) / (H * W)
    cgraph = softmax(cgraph / T, dim=0) * C
    cgraph = cgraph.reshape(N, C, 1, 1)
    return cgraph
