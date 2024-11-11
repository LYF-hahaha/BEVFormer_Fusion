# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
import matplotlib.pyplot as plt
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models import builder

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 pts_fusion_layer=None,
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        # 获取上一帧到当前帧的相对运动（用于计算prev_bev到当前帧的bev_grid相对运动，以进行temproal_self_attention）
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    # tmp_prev_bev = prev_bev[:, i].reshape(128, 128, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev,
                                          rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    # tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(128 * 128, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]    # can_bus info直接mlp，搞笑一样
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        # cam_info & level_info 学习权重 
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,     # Q
            feat_flatten,    # K
            feat_flatten,    # V
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,    # 前面can_bus算出来的帧间bev_grid相对运动
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                pts_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        img_bev_feature = self.get_bev_features(  # 这里是 encoder & decoder 一起
            mlvl_feats,                           # 还是会调用到上面无obj_que_emb的forward()  
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)
          
        # print('pre-fusion:')
        # exchange = 1024**3
        # a = (torch.cuda.memory_allocated())/exchange
        # b = (torch.cuda.memory_reserved())/exchange
        # print("  allocated:{:.2f}GB".format(a))
        # print("  reserved:{:.2f}GB".format(b))
        
        img_bev_feat_orig = img_bev_feature.clone().detach()
        
        bs = mlvl_feats[0].size(0)
        
        # img_bev_feature shape: (bs, bev_h*bev_w, embed_dims)即(1, 50x50, 256)
        # pts_feats shape: (1, 384, 128, 128)

        features = []
        # 点云BEV空间特征的尺寸为BEV_H_align, BEV_W_align（128x128）
        # BEV_H_align, BEV_W_align = pts_feats.shape[-2:]
        BEV_H_align, BEV_W_align = pts_feats[0].shape[-2:]

        # TODO: complete the following code with correct expressions.
        # 1. img_bev_feature由(bs, bev_h*bev_w, embed_dims)变换维度到 img_bev_feature_align：(bs, self.embed_dims, bev_h, bev_w)
        img_bev_feature = img_bev_feature.reshape(bs, bev_h, bev_w, self.embed_dims).permute(0,3,1,2)
        
        # 2. 将图像BEV特征通过interpolate插值函数插值到128x128大小，与点云BEV对齐，mode选择bilinear双线性插值法，设置对齐corners  scale_factor=(2.56,2.56)
        img_bev_feature = F.interpolate(img_bev_feature, (BEV_H_align, BEV_W_align), mode='bilinear', align_corners=True)
        
        # 3. 在features列表中添加上图像BEV特征和点云BEV特征
        features.append(img_bev_feature)
        features.append(pts_feats)
        
        # 4. 将features列表输入给融合层self.pts_fusion_layer做融合加强, [1, 384+256, 128, 128]-->[1, 256, 128, 128]
        bev_embed = self.pts_fusion_layer(features)
        
        # 5. 融合后的BEV特征bev_embed还原到初始形状, 
        # [bs, self.embed_dims, BEV_H_align, BEV_W_align]-->[bs, BEV_H_align*BEV_W_align, self.embed_dims]
        bev_embed = bev_embed.reshape(bs,self.embed_dims, BEV_H_align*BEV_W_align).permute(0,2,1)
        
        # object_query_embed在这里被分解成了query_pos和query
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        
        reference_points = self.reference_points(query_pos)   # nn.Linear
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        
        # print('post-fusion:')
        # exchange = 1024**3
        # a = (torch.cuda.memory_allocated())/exchange
        # b = (torch.cuda.memory_reserved())/exchange
        # print("  allocated:{:.2f}GB".format(a))
        # print("  reserved:{:.2f}GB".format(b))

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[BEV_H_align, BEV_W_align]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return img_bev_feat_orig, bev_embed, inter_states, init_reference_out, inter_references_out