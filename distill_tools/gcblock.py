import torch
import torch.nn as nn
from torch.nn.functional import softmax 
from torch.nn import LayerNorm, ReLU, Conv2d

class GCBlock(nn.Module):
    def __init__(self, chanel, r=1):
        # featurs r： reduction rate
        super(GCBlock, self).__init__()
        self.r = r  # 减少率
        C = chanel
        self.Cr = int(C/self.r)
        self.conv1 = Conv2d(C, 1, 1)
        self.conv2 = Conv2d(C, self.Cr, 1)
        self.conv3 = Conv2d(self.Cr, C, 1) 
        self.layerBN = LayerNorm([C, 1, 1])  
        self.relu = ReLU()

    def forward(self, featurs):
        imgs = featurs
        N, C, H, W = featurs.shape
        out = self.conv1(featurs).reshape(N, H*W, 1, 1)
        out = softmax(out, dim=0)
        out = torch.matmul(imgs.reshape(N, C, H*W), out.reshape(out.shape[:-1]))
        out = out.reshape(N, C, 1, 1)
        out = self.conv2(out)
        out = self.layerBN(out)
        out = self.relu(out)
        out = self.conv3(out)      
        return out + imgs