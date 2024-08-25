import torch.nn as nn
import torch.nn.functional as F
import torch
# from mmcv.cnn import constant_init, kaiming_init # gaidong
# from ..builder import DISTILL_LOSSES
# from skimage.metrics import structural_similarity as ssim
# from PIL import Image
import numpy as np
from mmengine.model import kaiming_init, constant_init

from mmdet.registry import MODELS

from .sed_compute import SED
import torchvision
import matplotlib.pyplot as plt
from .graph import *


@MODELS.register_module()
class TSKDLoss(nn.Module):
    """
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fg (float, optional): Weight of fg_loss. Defaults to 0.0005
        alpha_bg (float, optional): Weight of bg_loss. Defaults to 0.0005
        alpha_ins_rea (float, optional): Weight of ins_rea_loss. Defaults to 0.0005
        alpha_pixel_rea (float, optional): Weight of pixel_rea_loss. Defaults to 0.0005
        alpha_se (float, optional): Weight of alpha_se. Defaults to 0.0005

    """
    def __init__(self,
                 student_channels = 256,
                 teacher_channels = 256,
                 # name,
                 temp=0.5,
                 alpha_fg=0.0005,
                 alpha_bg=0.0005,
                 alpha_ins_rea=0.0005,
                 alpha_pixel_rea=0.0005,
                 alpha_se=0.0005,
                 loss_weight = 1,
                 sample_rate=2,
                 scale_rate=1,
                 roi_output_size=6
                 ):
        super(TSKDLoss, self).__init__()
        self.temp = temp
        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.alpha_ins_rea = alpha_ins_rea
        self.alpha_pixel_rea = alpha_pixel_rea
        self.alpha_se = alpha_se
        self.loss_weight = loss_weight

        self.sample_rate = sample_rate
        self.scale_rate = scale_rate
        self.roi_output_size = roi_output_size

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.reset_parameters()

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape

        S_attention_t, C_attention_t = self.enhanced_hybrid_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.enhanced_hybrid_attention(preds_S, self.temp)

        S_attention_t_resize = S_attention_t.unsqueeze(1)
        C_attention_t_resize = C_attention_t.unsqueeze(2).unsqueeze(3)
        enhanced_feature_T = preds_T * S_attention_t_resize * C_attention_t_resize
        enhanced_feature_T = enhanced_feature_T.mean(axis=1, keepdim=False).unsqueeze(1).unsqueeze(1)


        S_attention_s_resize = S_attention_s.unsqueeze(1)
        C_attention_s_resize = C_attention_s.unsqueeze(2).unsqueeze(3)
        enhanced_feature_S = preds_S * S_attention_s_resize * C_attention_s_resize
        enhanced_feature_S = enhanced_feature_S.mean(axis=1, keepdim=False).unsqueeze(1).unsqueeze(1)


        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)

        wmin, wmax, hmin, hmax = [], [], [], []
        gt_bboxes = [gt_bboxes[i].bboxes for i in range(len(gt_bboxes))]
        ins_feats_T_all = [torch.ones(1, len(gt_bboxes[i]), self.roi_output_size, self.roi_output_size) for i in range(N)]
        ins_feats_S_all = [torch.ones(1, len(gt_bboxes[i]), self.roi_output_size, self.roi_output_size) for i in range(N)]
        for i in range(N):

            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

                pooler = torchvision.ops.RoIAlign(output_size=self.roi_output_size, sampling_ratio=self.sample_rate,
                                                  spatial_scale=self.scale_rate)
                box = torch.tensor([[wmin[i][j], hmin[i][j], wmax[i][j] + 1, hmax[i][j] + 1]]).float().cuda()
                ins_feats_T_all[i][0][j] = (pooler(enhanced_feature_T[i], [box])).squeeze(0).squeeze(0)
                ins_feats_S_all[i][0][j] = (pooler(enhanced_feature_S[i], [box])).squeeze(0).squeeze(0)

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)  # if 真，0；假，1
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])
        ins_feats_T = torch.cat(ins_feats_T_all, 1).cuda()
        ins_feats_S = torch.cat(ins_feats_S_all, 1).cuda()

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        ins_rea_loss = self.get_ins_rela_loss(ins_feats_T, ins_feats_S)
        pixel_rea_loss = self.get_pixel_rela_loss(preds_S, preds_T)

        se_loss = self.get_sed_loss(preds_S, preds_T)

        loss = self.alpha_fg * fg_loss + self.alpha_bg * bg_loss
        + self.alpha_ins_rea * ins_rea_loss + self.alpha_pixel_rea * pixel_rea_loss + self.alpha_se * se_loss

        return loss

    def enhanced_hybrid_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape  # 2 256 7 7

        value = torch.abs(preds)
        # Bs*W*H 公式7
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)  # 2 7 7

        # Bs*C 公式8
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)  # 2 256
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention


    def get_ins_rela_loss(self, roi_T, roi_S):
        B, C, H, W = roi_T.shape
        roi_all_elements = roi_T.reshape(-1, 1)
        loss_mse = nn.MSELoss(reduction='sum')
        graph_Object = GloRe_Unit_2D(C, int(C * 0.5), normalize=True).cuda()
        graph_roi_rea_T = graph_Object(roi_T)
        graph_roi_rea_S = graph_Object(roi_S)
        return loss_mse(graph_roi_rea_T, graph_roi_rea_S) / len(roi_all_elements)


    def get_pixel_rela_loss(self, pred_T, pred_S):
        B, C, H, W = pred_T.shape
        pred_T_size = pred_T.reshape(-1, 1)
        loss_mse = nn.MSELoss(reduction='sum')
        graph_Object = GloRe_Unit_2D(C, int(C * 0.5), normalize=True).cuda()
        graph_pixel_rea_T = graph_Object(pred_T)
        graph_pixel_rea_S = graph_Object(pred_S)
        return loss_mse(graph_pixel_rea_T, graph_pixel_rea_S) / len(pred_T_size)

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss


    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context



    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

    #
    def get_sed_loss(self, x, y):

        sed = SED(11, True).cuda()
        sed_value = sed(x, y)
        sed_loss = (1 - sed_value) / 2
        return sed_loss
