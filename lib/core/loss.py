# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsDPDLoss(nn.Module):
    """
    Inspired from the DKD, decoupled the KL Loss in pose estimation. Try to figure out the process of failed KL Loss in pose task.
    """
    def __init__(self, use_target_weight=False, loss_weight=1., 
                 temperature=1., 
                 thre=0.5, episilon=1e-12):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight
        self.episilon = episilon
        self.thre = thre
        msg = 'JointsDPDLoss:\t' \
              'T: [{0}]\t' \
              'W: [{1}]\t' \
              'thre: [{2}]'.format(
                    self.temperature, self.loss_weight, self.thre
              )
        logger.info(msg)

    def _get_mask(self, target, thre=0.5):
        gt_mask = (target > thre).bool()
        other_mask = ~gt_mask
        return gt_mask.float(), other_mask.float()
    
    def _cat_mask(self, t, mask1, mask2):
        t1 = ( t * mask1).sum(dim=-1, keepdims=True)
        t2 = ( t * mask2).sum(dim=-1, keepdims=True)
        rt = torch.cat([t1, t2], dim=-1)
        return rt

    def dkl_loss(self, output, teacher, target):
        pred = output
        gt = target
        th = teacher

        gt_mask, other_mask = self._get_mask(gt, self.thre)

        logits_gt = F.softmax(gt/self.temperature, dim=-1)
        bi_gt = self._cat_mask(logits_gt, gt_mask, other_mask)
        gt_logits_part1 = F.softmax(gt/self.temperature - 1000*other_mask.float(), dim=-1)
        gt_logits_part2 = F.softmax(gt/self.temperature - 1000*gt_mask.float(), dim=-1)

        # Binary Keypoint Distillation (BiKD)
        pred_student = F.softmax(pred / self.temperature, dim=-1)
        pred_teacher = F.softmax(th / self.temperature, dim=-1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student + self.episilon)
        BiKD_loss = F.kl_div(log_pred_student, pred_teacher, reduction='none') * ( self.temperature ** 2)

        # Keypoint Area Distillation (KAD)
        pred_logits_part1 = F.softmax(pred / self.temperature - 1000 * other_mask.float(), dim=-1)
        th_logits_part1 = F.softmax(th / self.temperature - 1000 * other_mask.float(), dim=-1)
        log_pred_part1 = torch.log(pred_logits_part1 + self.episilon)
        KAD_loss = F.kl_div(log_pred_part1, th_logits_part1, reduction='none') * (self.temperature ** 2)

        # Non-keypoint Area Distillation (NAD)
        pred_logits_part2 = F.softmax(pred / self.temperature - 1000 * gt_mask.float(), dim=-1)
        th_logits_part2 = F.softmax(th / self.temperature - 1000 * gt_mask.float(), dim=-1)
        log_pred_part2 = torch.log(pred_logits_part2 + self.episilon)
        NAD_loss = F.kl_div(log_pred_part2, th_logits_part2, reduction='none') * (self.temperature ** 2)
        
        return {'BiKD':BiKD_loss.sum(-1).unsqueeze(-1), 'KAD':KAD_loss.sum(-1).unsqueeze(-1),       'NAD':NAD_loss.sum(-1).unsqueeze(-1)} 
                
    
    def forward(self, output, teacher, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_th = teacher.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = dict()
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_th = heatmaps_th[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss_ =  self.dkl_loss(
                    heatmap_pred.mul(target_weight[:, idx]), heatmap_th.mul(target_weight[:, idx]), 
                    heatmap_gt.mul(target_weight[:, idx])
                )
                
            else:
                loss_ =  self.dkl_loss(heatmap_pred, heatmap_th, heatmap_gt)

            for key in loss_.keys():
                if key in loss.keys():
                    loss[key] = torch.cat((loss[key],loss_[key]),1)
                else:
                    loss[key] = loss_[key]
        return loss