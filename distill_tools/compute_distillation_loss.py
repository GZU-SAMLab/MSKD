import torch
from distill_tools.bmask import bmask
from distill_tools.smask import smask
from distill_tools.plane_mask import plane_mask
from distill_tools.channel_mask import channel_mask
import torch.nn as nn
from distill_tools.gcblock import GCBlock



def distillation_loss(target, \
        head_distillation=True, \
        neck_distillation=True, \
        backbone_distillation=True, \
        student_head_feature=[], \
        student_neck_feature=[], \
        student_backbone_feature=[], \
        teacher_head_feature=[], \
        teacher_neck_feature=[], \
        teacher_backbone_feature=[],\
        head_weight=1.0, \
        neck_weight=1.0, \
        backbone_weight=1.0, device='cpu', \
        target_weight=0.5, \
        background_weight=0.5, \
        attention_weight=0.5, \
        global_weight=0.5):

    # DISTILLATION_RATE = 1e-9
    DISTILLATION_RATE = 1.0

    # head loss
    # focal loss
    h_local_loss = torch.zeros(1).to(device)
    # global loss
    h_global_loss = torch.zeros(1).to(device)
    # attention loss
    h_attention_loss = torch.zeros(1).to(device)

    # neck loss
    n_local_loss = torch.zeros(1).to(device)
    # global loss
    n_global_loss = torch.zeros(1).to(device)
    # attention loss
    n_attention_loss = torch.zeros(1).to(device)
 
    # backbone loss
    b_local_loss = torch.zeros(1).to(device)
    # global loss
    b_global_loss = torch.zeros(1).to(device)
    # attention loss
    b_attention_loss = torch.zeros(1).to(device)

    # gcblock
    head_gcbolck = [GCBlock(f.shape[1]).to(device) for f in teacher_head_feature]
    neck_gcbolck = [GCBlock(f.shape[1]).to(device) for f in teacher_neck_feature]
    backbone_gcbolck = [GCBlock(f.shape[1]).to(device) for f in teacher_backbone_feature]

    l1 = nn.L1Loss()

    for i, (tbf_o, sbf_o, tnf_o, snf_o, thf_o, shf_o) in enumerate(zip(\
            teacher_backbone_feature, student_backbone_feature,\
            teacher_neck_feature, student_neck_feature, \
            teacher_head_feature, student_head_feature)):
        tbf = tbf_o.sigmoid()
        sbf = sbf_o.sigmoid()
        tnf = tnf_o.sigmoid()
        snf = snf_o.sigmoid()
        thf = thf_o.sigmoid()
        shf = shf_o.sigmoid()
        # head
        if head_distillation:
            tacmask = channel_mask(thf)  # channel attention mask
            sacmask = channel_mask(shf)
            tapmask = plane_mask(thf)   # plane attention mask
            sapmask = plane_mask(shf)
            tsmask = smask(thf, target) # scale mask
            # ssmask = smask(shf, target)
            tbmask = bmask(thf, target) # binary mask
            # sbmask = bmask(shf, target)
            tg_feature = head_gcbolck[i](thf)  # global loss
            sg_feature = head_gcbolck[i](shf)
            h_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (thf - shf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (thf - shf)**2) * background_weight)/thf.numel()
            h_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask))/thf.shape[0] * attention_weight
            h_global_loss += torch.sum((tg_feature - sg_feature)**2) / thf.numel() * global_weight

        if neck_distillation:
            tacmask = channel_mask(tnf)  # channel attention mask
            sacmask = channel_mask(snf)
            tapmask = plane_mask(tnf)   # plane attention mask
            sapmask = plane_mask(snf)
            tsmask = smask(tnf, target) # scale mask
            # ssmask = smask(shf)
            tbmask = bmask(tnf, target) # binary mask
            # sbmask = bmask(shf)
            tg_feature = neck_gcbolck[i](tnf)  # global loss
            sg_feature = neck_gcbolck[i](snf)
            n_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (tnf - snf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (tnf - snf)**2) * backbone_weight)/tnf.numel()
            n_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask))/thf.shape[0] * attention_weight
            n_global_loss += torch.sum((tg_feature - sg_feature)**2) / tnf.numel() * global_weight
        
        if backbone_distillation:
            tacmask = channel_mask(tbf)  # channel attention mask
            sacmask = channel_mask(sbf)
            tapmask = plane_mask(tbf)   # plane attention mask
            sapmask = plane_mask(sbf)
            tsmask = smask(tbf, target) # scale mask
            # ssmask = smask(shf)
            tbmask = bmask(tbf, target) # binary mask
            # sbmask = bmask(shf)
            tg_feature = backbone_gcbolck[i](tbf)  # global loss
            sg_feature = backbone_gcbolck[i](sbf)
            b_local_loss += (torch.sum(\
                tbmask * tsmask * tacmask * tapmask * (tbf - sbf)**2) * target_weight +\
                torch.sum((1-tbmask) * tsmask * tacmask * tapmask * (tbf - sbf)**2) * backbone_weight) / tbf.numel()
            b_attention_loss += (l1(tapmask, sapmask) + l1(tacmask, sacmask)) /thf.shape[0]* attention_weight
            b_global_loss += torch.sum((tg_feature - sg_feature)**2) /tbf.numel() * global_weight
    return ((h_local_loss + h_attention_loss + h_global_loss)*head_weight + \
                (n_local_loss + n_attention_loss + n_global_loss)*neck_weight + \
                    (b_local_loss + b_attention_loss + b_global_loss)*backbone_weight) * DISTILLATION_RATE,\
                        (h_local_loss + h_attention_loss + h_global_loss)*head_weight,\
                            (n_local_loss + n_attention_loss + n_global_loss)*neck_weight,\
                                (b_local_loss + b_attention_loss + b_global_loss)*backbone_weight
