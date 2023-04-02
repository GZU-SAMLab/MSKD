import torch
from torch.nn.functional import softmax

# 获取空间注意力掩码
def plane_mask(featurs, T=1):
    N, C, H, W = featurs.shape
    sgraph = torch.abs(featurs).sum(dim=1) / C
    sgraph = softmax(sgraph / T, dim=0) * H * W
    sgraph = sgraph.reshape(N, 1, H, W)
    return sgraph