from torch.nn.functional import softmax
import torch

def nh_soft_loss(tf, sf, T=3, xy_w=1.0, wh_w=1.0, conf_w=1e-2, cls_w=1.0):
    xy_loss = 0.0
    wh_loss = 0.0
    conf_loss= 0.0
    cls_loss=0.0
    for t,s in zip(tf,sf):
        t = t.reshape(-1,t.shape[-1])
        s = s.reshape(-1,s.shape[-1])
        t = t/T
        s = s/T
        xy_loss += ((t[:,0] - s[:,0])**2 + (t[:,1] - s[:,1])**2).sum() \
                        / t.shape[0]
        wh_loss += (((torch.sqrt(torch.exp(t[:,2])) - torch.sqrt(torch.exp(s[:,2]))))**2 + 
                        (torch.sqrt(torch.exp(t[:,2])) - torch.sqrt(torch.exp(s[:,2])))**2).sum() \
                            / t.shape[0]
        conf_loss += ((t[:,4] - s[:,4])**2).sum() / t.shape[0]
        cls_loss += ((softmax(t[:,5:], dim=1) - softmax(s[:,5:], dim=1))**2).sum() / t.shape[0]

    return (xy_w * xy_loss + wh_w * wh_loss +  conf_w * conf_loss + cls_w * cls_loss) / 4