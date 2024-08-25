# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import (InstanceList, OptInstanceList, OptConfigType, reduce_mean)
from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


@MODELS.register_module()
class TSKDDDOD(CrossKDSingleStageDetector):

    def __init__(self, 
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg,**kwargs)
        self.loss_iou_kd = None
        if kd_cfg.get('loss_iou_kd', None):
            self.loss_iou_kd = MODELS.build(kd_cfg['loss_iou_kd'])
                
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_iou, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        tea_x,
                        self.teacher.bbox_head.scales, 
                        module=self.teacher)
            
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_iou, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        stu_x,
                        self.bbox_head.scales, 
                        module=self)
            
        reused_cls_scores, reused_bbox_preds, reused_iou = multi_apply(
            self.reuse_teacher_head, 
            tea_cls_hold, 
            tea_reg_hold, 
            stu_cls_hold,
            stu_reg_hold, 
            self.teacher.bbox_head.scales)


        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_cls_scores,
                                   tea_bbox_preds,
                                   tea_iou,
                                   tea_x,
                                   stu_cls_scores,
                                   stu_bbox_preds,
                                   stu_iou,
                                   stu_x,
                                   reused_cls_scores,
                                   reused_bbox_preds,
                                   reused_iou,
                                   batch_gt_instances,
                                   batch_img_metas, 
                                   batch_gt_instances_ignore)
        return losses
    
    def forward_hkd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.atss_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.atss_reg(reg_feat)).float()
        iou = module.bbox_head.atss_iou(reg_feat)
        return cls_score, bbox_pred, iou, cls_feat_hold, reg_feat_hold
    
    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.atss_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.atss_reg(reused_reg_feat)).float()
        reused_iou = module.atss_iou(reused_reg_feat)
        return reused_cls_score, reused_bbox_pred, reused_iou
    
    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_iou: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            iou_preds: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            reused_iou: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        targets_com = self.bbox_head.process_predictions_and_anchors(
            anchor_list, valid_flag_list, cls_scores, bbox_preds,
            batch_img_metas, batch_gt_instances_ignore)

        (anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list,
         bbox_pred_list, batch_gt_instances_ignore) = targets_com

        cls_targets = self.bbox_head.get_cls_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (cls_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
        avg_factor = max(avg_factor, 1.0)

        reweight_factor_per_level = self.bbox_head.calc_reweight_factor(labels_list)

        cls_losses_cls, = multi_apply(
            self.bbox_head.loss_cls_by_feat_single,
            cls_scores,
            labels_list,
            label_weights_list,
            reweight_factor_per_level,
            avg_factor=avg_factor)

        reg_targets = self.bbox_head.get_reg_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (reg_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
        avg_factor = max(avg_factor, 1.0)

        reweight_factor_per_level = self.bbox_head.calc_reweight_factor(labels_list)

        reg_losses_bbox, reg_losses_iou = multi_apply(
            self.bbox_head.loss_reg_by_feat_single,
            reg_anchor_list,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            reweight_factor_per_level,
            avg_factor=avg_factor)

        losses = dict(loss_cls=cls_losses_cls, loss_bbox=reg_losses_bbox, loss_iou=reg_losses_iou)
        # cls_reg_targets = self.bbox_head.get_targets(
        #     anchor_list,
        #     valid_flag_list,
        #     batch_gt_instances,
        #     batch_img_metas,
        #     batch_gt_instances_ignore=batch_gt_instances_ignore)
        #
        # (anchor_list, labels_list, label_weights_list, bbox_targets_list,
        #  bbox_weights_list, avg_factor) = cls_reg_targets

        # avg_factor = reduce_mean(
        #     torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
        #
        # losses_cls, losses_bbox, loss_iou, \
        #     bbox_avg_factor = multi_apply(
        #         self.bbox_head.loss,
        #         anchor_list,
        #         cls_scores,
        #         bbox_preds,
        #         iou,
        #         labels_list,
        #         label_weights_list,
        #         bbox_targets_list,
        #         avg_factor=avg_factor)
        #
        # bbox_avg_factor = sum(bbox_avg_factor)
        # bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        # losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        # losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=loss_iou)

        losses_cls_kd, losses_reg_kd, losses_iou_kd = multi_apply(
            self.pred_imitation_loss_single,
            labels_list,
            reg_anchor_list,
            tea_cls_scores,
            tea_bbox_preds,
            tea_iou,
            reused_cls_scores,
            reused_bbox_preds,
            reused_iou,
            label_weights_list,
            avg_factor=avg_factor)
        losses.update(dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd, losses_iou_kd=losses_iou_kd))
        
        # if self.with_feat_distill:
        #     losses_feat_kd = [
        #         self.loss_feat_kd(feat, tea_feat)
        #         for feat, tea_feat in zip(feats, tea_feats)
        #     ]
        #     losses.update(loss_feat_kd=losses_feat_kd)
        # return losses
        # assert batch_img_metas == batch_img_metas
        if self.with_feat_distill:
            # losses_feat_kd = [
            #     self.loss_feat_kd(feat, tea_feat)
            #     for feat, tea_feat in zip(feats, tea_feats)
            # ]
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat, batch_gt_instances, batch_img_metas)
                for feat, tea_feat in zip(feats, tea_feats)
            ]

            for i, loss in enumerate(losses_feat_kd):
                losses.update({"loss_feat_kd_{}".format(i): loss})
        return losses
    
    def pred_imitation_loss_single(self, 
                                   labels,
                                   anchors,
                                   tea_cls_score, 
                                   tea_bbox_pred,
                                   tea_iou,
                                   reused_cls_score, 
                                   reused_bbox_pred,
                                   reused_iou,
                                   label_weights, 
                                   avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # regression branch distillation
        bbox_coder = self.bbox_head.bbox_coder
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        anchors = anchors.reshape(-1, anchors.size(-1))
        tea_bbox_pred = bbox_coder.decode(anchors, tea_bbox_pred)
        reused_bbox_pred = bbox_coder.decode(anchors, reused_bbox_pred)
        
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights,
            avg_factor=avg_factor)
        
        # centernesses branch distillation
        labels = labels.reshape(-1)
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        tea_iou = tea_iou.permute(0, 2, 3, 1).reshape(-1)
        reused_iou = reused_iou.permute(0, 2, 3, 1).reshape(-1)

        if len(pos_inds) > 0:
            loss_iou_kd = self.loss_iou_kd(
                reused_iou[pos_inds],
                tea_iou[pos_inds].sigmoid(),
                avg_factor=avg_factor)
        else:
            loss_iou_kd = reused_iou.new_tensor(0.)
        return loss_cls_kd, loss_reg_kd, loss_iou_kd