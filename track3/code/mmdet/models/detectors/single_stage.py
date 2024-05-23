# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .tSF import tSF

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff

def dist2_masked(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None, replay_data=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    if attention_mask!=None:
        diff = diff * attention_mask
    if channel_attention_mask!=None:
        diff = diff * channel_attention_mask
    if replay_data!=None:
        for idx in range(len(replay_data)):
            if replay_data[idx] == 0:
                diff[idx] = diff[idx] * 0.000001
    diff = torch.sum(diff) ** 0.5
    return diff

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.adaptation_type = '1x1conv'
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        '''
        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ])
        if self.adaptation_type == '1x1conv':
            #   1x1 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            ])

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        '''

        # FPN kd
        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

        '''
        self.vfnet_head_kd = False
        if self.vfnet_head_kd:
            # head kd
            self.student_non_local_cls = nn.ModuleList(
                [
                    NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                    NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                    NonLocalBlockND(in_channels=256),
                    NonLocalBlockND(in_channels=256),
                    NonLocalBlockND(in_channels=256)
                ]
            )
            self.teacher_non_local_cls = nn.ModuleList(
                [
                    NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                    NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                    NonLocalBlockND(in_channels=256),
                    NonLocalBlockND(in_channels=256),
                    NonLocalBlockND(in_channels=256)
                ]
            )
            self.non_local_adaptation_cls = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            ])
        '''

        '''
        # FPN tSF
        self.fpn_tSF = nn.ModuleList(
            [
                tSF(feature_dim=256, num_queries=5, num_heads=4, FFN_method='MLP'),
                tSF(feature_dim=256, num_queries=5, num_heads=4, FFN_method='MLP'),
                tSF(feature_dim=256, num_queries=5, num_heads=4, FFN_method='MLP'),
                tSF(feature_dim=256, num_queries=5, num_heads=4, FFN_method='MLP'),
                tSF(feature_dim=256, num_queries=5, num_heads=4, FFN_method='MLP')
            ]
        )
        '''
        '''
        # FPN nonlocal
        self.fpn_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        '''
        '''
        ############### loading EGO json annotations for kd #######################  
        DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
        # exp0
        train_anno_path_exp0 = DATASET_PATH + '/' + 'ego_objects_challenge_trainSplit_submit_exp0.json'
        print(f"Loading {train_anno_path_exp0}")
        train_anno_exp0 = _load_json(train_anno_path_exp0)
        train_group_id_exp0 = []
        for img_anno in train_anno_exp0["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id_exp0.append(img_group_id)
        self.train_group_id_unique_exp0 = list(set(train_group_id_exp0))
        # exp1
        train_anno_path_exp1 = DATASET_PATH + '/' + 'ego_objects_challenge_trainSplit_submit_exp1.json'
        print(f"Loading {train_anno_path_exp1}")
        train_anno_exp1 = _load_json(train_anno_path_exp1)
        train_group_id_exp1 = []
        for img_anno in train_anno_exp1["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id_exp1.append(img_group_id)
        self.train_group_id_unique_exp1 = list(set(train_group_id_exp1))
        # exp2
        train_anno_path_exp2 = DATASET_PATH + '/' + 'ego_objects_challenge_trainSplit_submit_exp2.json'
        print(f"Loading {train_anno_path_exp2}")
        train_anno_exp2 = _load_json(train_anno_path_exp2)
        train_group_id_exp2 = []
        for img_anno in train_anno_exp2["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id_exp2.append(img_group_id)
        self.train_group_id_unique_exp2 = list(set(train_group_id_exp2))
        # exp3
        train_anno_path_exp3 = DATASET_PATH + '/' + 'ego_objects_challenge_trainSplit_submit_exp3.json'
        print(f"Loading {train_anno_path_exp3}")
        train_anno_exp3 = _load_json(train_anno_path_exp3)
        train_group_id_exp3 = []
        for img_anno in train_anno_exp3["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id_exp3.append(img_group_id)
        self.train_group_id_unique_exp3 = list(set(train_group_id_exp3))
        # exp4
        train_anno_path_exp4 = DATASET_PATH + '/' + 'ego_objects_challenge_trainSplit_submit_exp4.json'
        print(f"Loading {train_anno_path_exp4}")
        train_anno_exp4 = _load_json(train_anno_path_exp4)
        train_group_id_exp4 = []
        for img_anno in train_anno_exp4["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id_exp4.append(img_group_id)
        self.train_group_id_unique_exp4 = list(set(train_group_id_exp4))
        '''

        '''
        ############### loading EGO json annotations for scene classification #######################  
        DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
        train_anno_path = DATASET_PATH + '/' + 'ego_objects_challenge_train.json'
        print(f"Loading {train_anno_path}")
        train_anno = _load_json(train_anno_path)
        train_group_id = []
        for img_anno in train_anno["images"]:
            img_group_id = img_anno["group_id"]
            train_group_id.append(img_group_id)
        self.train_group_id_unique = list(set(train_group_id))
        # scene classification loss
        self.scene_num = len(self.train_group_id_unique)
        self.x_H = 1
        self.x_W = 1
        self.scene_classifier = nn.Linear(self.x_H*self.x_W*2048, self.scene_num)
        '''

    def extract_feat(self, img, backbone_feat=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if backbone_feat:
            backbone_x = x
        if self.with_neck:
            x = self.neck(x)
            '''
            # FPN tSF
            x_out = []
            for _i in range(len(x)):
                x_tsf, _ = self.fpn_tSF[_i](x[_i])
                x_out.append(x_tsf)
            x = tuple(x_out)
            '''
            '''
            # FPN nonlocal
            x_out = []
            for _i in range(len(x)):
                x_out.append(self.fpn_non_local[_i](x[_i]))
            x = tuple(x_out)
            '''
        if backbone_feat:
            return x, backbone_x
        else:
            return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def get_teacher_info(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        xs = self.extract_feat(img)
        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
        t_info = {'feat': xs[-1]}
        '''
        if self.training and self.vfnet_head_kd:
            vfnet_head_outs = self.bbox_head.forward(xs[-1])
            cls_score, bbox_pred, bbox_pred_refine, cls_feat = vfnet_head_outs[0], vfnet_head_outs[1], vfnet_head_outs[2], vfnet_head_outs[3]
            t_info = {
                'feat':xs[-1],
                'cls_score':cls_score,
                'bbox_pred':bbox_pred,
                'bbox_pred_refine':bbox_pred_refine,
                'cls_feat':cls_feat}
        else:
            t_info = {'feat': xs[-1]}
        '''
        return t_info

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      exp_id=None,
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        xs = self.extract_feat(img)
        #xs, backbone_xs = self.extract_feat(img, backbone_feat=True)
        #print(f'backbone_xs size={backbone_xs[-1].size()}')
        loss_weights = None
        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
        elif loss_weights is None:
            loss_weights = [0.5] + [1]*(len(xs)-1)  # Reference CBNet paper

        def upd_loss(losses, idx, weight):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if weight != 1 and 'loss' in k:
                    new_k = '{}_w{}'.format(new_k, weight)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        # loss for head
        losses = dict()
        for i,x in enumerate(xs):
            box_losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore)
            if len(xs) > 1:
                box_losses = upd_loss(box_losses, idx=i, weight=loss_weights[i])
            losses.update(box_losses)

        # loss for kd
        x = xs[-1]
        '''
        if t_info is not None:
            # kd
            t = 0.1
            s_ratio = 1.0
            kd_feat_loss = 0
            kd_channel_loss = 0
            kd_spatial_loss = 0

            #   for channel attention
            c_t = 0.1
            c_s_ratio = 1.0
            t_feats = t_info['feat']
            for _i in range(len(t_feats)):
                t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
                size = t_attention_mask.size()
                t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
                t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                t_attention_mask = t_attention_mask.view(size)

                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                s_attention_mask = s_attention_mask.view(size)

                c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_t_attention_mask.size()
                c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
                c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_s_attention_mask.size()
                c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
                c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                sum_attention_mask = sum_attention_mask.detach()

                c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask = c_sum_attention_mask.detach()

                kd_feat_loss += dist2_masked(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                                    channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6 * 1
                kd_channel_loss += dist2_masked(torch.mean(t_feats[_i], [2, 3]),
                                            self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6 * 1
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                            x[_i].size(3))
                kd_spatial_loss += dist2_masked(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6 * 1

            kd_feat_loss_dict = {'kd_feat_loss': kd_feat_loss}
            kd_channel_loss_dict = {'kd_channel_loss': kd_channel_loss}
            kd_spatial_loss_dict = {'kd_spatial_loss': kd_spatial_loss}
            losses.update(kd_feat_loss_dict)
            losses.update(kd_channel_loss_dict)
            losses.update(kd_spatial_loss_dict)
        '''
        
        if t_info is not None:
            kd_nonlocal_loss = 0
            t_feats = t_info['feat']
            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                #   print(s_relation.size())
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
            kd_nonlocal_loss = kd_nonlocal_loss * 7e-5 * 6 * 1
            kd_nonlocal_loss_dict = {'kd_nonlocal_loss': kd_nonlocal_loss}
            losses.update(kd_nonlocal_loss_dict)

            '''
            # vfnet head kd
            if self.vfnet_head_kd:
                # teacher
                t_cls_score = t_info['cls_score']
                t_bbox_pred = t_info['bbox_pred']
                t_bbox_pred_refine = t_info['bbox_pred_refine']
                t_cls_feat = t_info['cls_feat']
                # student
                vfnet_head_outs = self.bbox_head.forward(x)
                cls_score, bbox_pred, bbox_pred_refine, cls_feat = vfnet_head_outs[0], vfnet_head_outs[1], vfnet_head_outs[2], vfnet_head_outs[3]
                # cls_score_kd
                kd_cls_score_loss = 0
                for _i in range(len(t_cls_score)):
                    kd_cls_score_loss += torch.dist(cls_score[_i], t_cls_score[_i], p=2)
                losses.update(kd_cls_score_loss=kd_cls_score_loss * 7e-5 * 6 * 1)
                # cls_feat_kd
                kd_cls_feat_nonlocal_loss = 0
                for _i in range(len(t_cls_feat)):
                    s_relation = self.student_non_local_cls[_i](cls_feat[_i])
                    t_relation = self.teacher_non_local_cls[_i](t_cls_feat[_i])
                    kd_cls_feat_nonlocal_loss += torch.dist(self.non_local_adaptation_cls[_i](s_relation), t_relation, p=2)
                losses.update(kd_cls_feat_nonlocal_loss=kd_cls_feat_nonlocal_loss * 7e-5 * 6 * 1)
            '''

        '''
        # scene classification loss
        batch_group_id_label = []
        for img_meta in img_metas:
            img_path = img_meta['filename']
            img_name = img_path.split("/")[-1]
            group_id = img_name.split("_")[0]
            #print(f'group_id={group_id}')
            batch_group_id_label.append(self.train_group_id_unique.index(group_id))
        batch_group_id_label = torch.tensor(batch_group_id_label).cuda()
        #print(f'batch_group_id_label={batch_group_id_label}')
        #x_out = nn.AdaptiveAvgPool2d((self.x_H, self.x_W))(x[-1]) # P7
        x_out = nn.AdaptiveAvgPool2d((self.x_H, self.x_W))(backbone_x[-1]) # C5
        #print(f'x_out size={x_out.size()}')
        scene_cls_scores = self.scene_classifier(x_out.view(x_out.size(0), -1))
        scene_cls_loss = F.cross_entropy(scene_cls_scores, batch_group_id_label)
        losses.update(scene_cls_loss=scene_cls_loss * 1)
        '''

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
