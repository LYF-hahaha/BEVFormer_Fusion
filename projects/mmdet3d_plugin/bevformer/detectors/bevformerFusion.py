# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import matplotlib.pyplot as plt
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormerFusion(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormerFusion,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        # self.voxelize_reduce = True


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        # TODO 1: complete the following code with correct expressions.
        # 参考mmdet3d/models/detectors/centerpoint.py中的extract_pts_feat点云特征提取部分补全以下代码

        voxel_features = self.pts_voxel_encoder(voxels,
                                                num_points,
                                                coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, 
                                    coors,
                                    batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
        # 该部分内容参考自mmdet3d/models/detectors/mvx_two_stage.py
        # voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
        #                                         voxel_dict['num_points'],
        #                                         voxel_dict['coors'], 
        #                                         img_feats,
        #                                         batch_input_metas)
        # batch_size = voxel_dict['coors'][-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, 
        #                             voxel_dict['coors'],
        #                             batch_size)
        # x = self.pts_backbone(x)
        # if self.with_pts_neck:
        #     x = self.pts_neck(x)
        # return x
    
        # raise NotImplementedError

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            # Returns:
            #   voxels: [M, max_points, ndim] float tensor. only contain points and returned when max_points != -1.
            #          M:体素个数、max_points:一个体素内点的最大值、ndim:一个点的维度
            #   coordinates: [M, 3] int32 tensor, always returned.
            #          体素化的坐标研究范围，可视化后是平面的
            #   num_points_per_voxel: [M] int32 tensor. Only returned when max_points != -1.
            #          体素中点的个数（维度是[29882]，每个元素是1~20之间的数，表示某个体素中点的数量）              
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)       
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        return voxels, num_points, coors_batch
    

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, points, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if points:
            pts_feats = self.extract_pts_feat(points)

        return (img_feats, pts_feats)


    def forward_pts_train(self,
                          img_feats,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            img_feats, pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, points, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape     # 不止拿一张照片,多张提升数据可靠性,在tiny里是2
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, points=points, len_queue=len_queue)[0]  # 这后面一个[0]把pts_feat给去了
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, points, img_metas, prev_bev, only_bev=True)  # 第一帧prev_bev=None, 第一帧encoder(没用decoder)输出的bev_embed就是第二针的prev_bev
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, points, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        # img_feats = self.extract_feat(img=img, img_metas=img_metas)
        img_feats, pts_feats = self.extract_feat(img=img, points=points, img_metas=img_metas)
        # img_feats = torch.Size([1, 6, 256, 15, 25])
        # pts_feats = torch.Size([1, 384, 128, 128])
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # 注意，这里把上一帧的img_bev_feats赋值给new_prev_bev了
        # 而返回的原先new_prev_bev (即带pts的fusion_embed丢弃了)
        new_prev_bev, _, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results


    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""

        # TODO 2: replace the 'None' values in the following code with correct expressions.
        # 调用self.extract_feat函数，输入图像img，点云points和图像信息img_metas，输出图像特征和点云特征
        img_feats, pts_feats = self.extract_feat(img, points, img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        img_bev_feature, new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, pts_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return img_bev_feature, new_prev_bev, bbox_list


    def simple_test_pts(self, img_feats, pts_feats, img_metas, prev_bev=None, rescale=False):
        """Test function"""

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas, prev_bev=prev_bev)
        
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['img_bev_feature'], outs['bev_embed'], bbox_results

    
        # # 可视化BEV融合特征
        # new_prev_bev_np = new_prev_bev.permute(1,0,2)
        # new_prev_bev_np = new_prev_bev_np.view(1,128,128,256)
        # new_prev_bev_np = new_prev_bev_np.cpu().numpy()
        # bev_feat_abs = np.abs(new_prev_bev_np)
        # # Calculate the mean across all channels
        # bev_feat_mean = np.mean(bev_feat_abs, axis=3)
        # bev_feat_mean = np.clip(bev_feat_mean,0,1.0)
        # # Visualize the result
        # plt.imshow(bev_feat_mean[0], cmap='viridis')
        # # plt.colorbar()
        # plt.axis('off')
        # name = img_metas[0]['pts_filename'].split('__')[-1]
        # plt.savefig(f'vis_bev_feat/fusion_bev/{name}.png')
        
        # # 可视化图像BEV特征
        # img_bev_feature_np = img_bev_feature.view(1, 50, 50, 256)
        # img_bev_feature_np = img_bev_feature_np.cpu().numpy()
        # bev_feat_abs = np.abs(img_bev_feature_np)
        # # Calculate the mean across all channels
        # bev_feat_mean = np.mean(bev_feat_abs, axis=3)
        # bev_feat_mean = np.clip(bev_feat_mean,0,1.0)
        # # Visualize the result
        # plt.imshow(bev_feat_mean[0], cmap='viridis')
        # # plt.colorbar()
        # plt.axis('off')
        # name = img_metas[0]['pts_filename'].split('__')[-1]
        # plt.savefig(f'vis_bev_feat/img_bev/{name}.png')

        # # 可视化点云BEV特征
        # pts_feats_np = pts_feats.cpu().numpy()
        # bev_feat_abs = np.abs(pts_feats_np)
        # # Calculate the mean across all channels
        # bev_feat_mean = np.mean(bev_feat_abs, axis=1)
        # bev_feat_mean = np.clip(bev_feat_mean,0,1.0)
        # # Visualize the result
        # plt.imshow(bev_feat_mean[0], cmap='viridis')
        # # plt.colorbar()
        # plt.axis('off')
        # name = img_metas[0]['pts_filename'].split('__')[-1]
        # plt.savefig(f'vis_bev_feat/pts_bev/{name}.png')

        # return img_bev_feature, new_prev_bev, bbox_list