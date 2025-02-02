
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from . import guide_pts_gen as gpg
import os
import pickle
from mmdet3d.models import builder
from torch.nn import functional as F


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            # H=W=150 z=8 num_pts_in_pillar=4
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, # [0.5, 2.83, 5.17 ,7.5]
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z #  0.0625~0.9375  X,Y向拓展成H,W. Z向还是num_pts_in-pillar
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W # 0.0033~0.9967
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # shape=[1,4,22500,3] 
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d  # 这里返回的也是ratio

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  guide_3d_curt, img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        
        reference_points = reference_points.clone()
        # 参考点的值从pillar空间转换成实际空间坐标值(本来是 x,y:0.0033~0.9967 z:0.0625~0.9375)
        # 变换到点云的范围内. 这也是为何get_reference_points中会/H, /W, /Z, 先化到[0, 1]变成ratio.
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        # TODO: 
        # 1、reshape成(150,150) 加入guide_pts
        # 2、加多少个点就在边界减多少个点，使每层的总数为22500
        # 3、flatten
        guide_num = guide_3d_curt.shape[0]
        guide_points = torch.from_numpy(guide_3d_curt)
        
        reference_points[0,0,:guide_num,:]=guide_points
        # device = torch.device("cuda")
        # reference_points = reference_points.to(device)
        
        # 由(x, y, z) 变成(x, y, z, 1) 便于与4*4的参数矩阵相乘
        # ref_pts shape变化:[1,4,22500,3] → [1, 4, 22500, 4]
        reference_points = torch.cat((reference_points,
                                      torch.ones_like(reference_points[..., :1])),  -1)
        # 此时reference_points可以当成是点云的点了
        
        # num_query等于H*W*Z. 等于grid_points的数量
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)  # lidar2img.shape=[1,6,4,4]
        
        # 要往每个相机上去投影. 因此先申请num_cam份, 每个cam_view下都有一个四方四正的init_grid ref-3d-pts
        # reference_points的shape就变成了, (D, b, num_cam, num_query, 4, 1) 便于和4*4的矩阵做matmul
        # 变换后的shape: [4, 1, 6, 22500, 4, 1]
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        # 一起有6个变换矩阵
        # 相机参数由(b,num_cam, 4, 4) 变成(1, b, num_cam, 1, 4, 4) 再变成(D,b,num_cam,num_query,4,4)
        # 变换后的shape: lidar2img.shape [4, 1, 6, 22500, 4, 4]
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1) 
        
        # 一起投影至各cam下 z_layer, bs, cam_num, query_num, coords&ones=[4, 1, 6, 22500, 4]
        # 这一步的投影，成像平面是无限大的，所以才会出现能斜着看到ref_3d的4层边界的情况
        # 计算后的shape:[4, 1, 6, 22500, 4]
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        
        # 标记出相机前和相机后的点 (因为相机后面的点投过来之后第三位是负的，所以>eps)
        eps = 1e-5
        # 计算后的shape: [4, 1, 6, 22500, 1]
        bev_mask = (reference_points_cam[..., 2:3] > eps)  # [4,2,6,22500,1] [True] & [False]
        
        # 再做齐次化. 得到像素坐标 (uZ,vZ,Z) → (u,v,1)
        # 如果是在像素平面前的点，就归一化；如果是在平面后的点，就变得巨大
        # torch.maximum: 返回一个tensor，每个位置都是输入的两个变量中对应位置更大的那个变量
        # 这就已经只剩[u,v]了
        # 计算后的shape: [4, 1, 6, 22500, 2]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)  # ones_like: 全是1

        # 由像素坐标转成相对于图像的ratio
        # NOTE 这里如果不同相机size不一样的话，要除以对应的相机的size
        # 计算后的shape:
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        # 再把超出图像fov范围的点给去掉
        # 上一步中，ref_pts_cam已经是相对于像素平面的ratio了，大于1和小于0的都是超范围FOV的
        # 计算后的shape:[4, 1, 6, 22500, 1]
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)   # 用0替代nan
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))   # 用0替代nan 

        # 由(D, b, num_cam, num_query, 2) 变成 (num_cam, b, num_query, D, 2)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        # ref_pts_cam计算后的shape: [6, 1, 22500, 4, 2]
        # bev_mask计算后的shape: [6, 1, 22500, 4]
        
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        # 至此 reference_points_cam代表的就是像素点相对于各个相机的ratio
        # bev_mask就代表哪些点是有效的
        return reference_points_cam, bev_mask

    def guide_obtain(self, img_metas):

        # 当前帧和上一帧点云路径获取
        pts_curt_path =  img_metas[0]['pts_filename']
        tmp = pts_curt_path.split("/")
        tk_pts_info_path = os.path.join(tmp[0],tmp[1],tmp[2],'tk_pts_info.pkl')
        f1 = open(tk_pts_info_path, 'rb')
        tk_pts = pickle.load(f1)
        # 上一帧不存在的话，就用当前帧=上一帧
        if len(img_metas[0]['prev_idx']) == 0:
            pts_lidar_prev = np.fromfile(pts_curt_path, dtype=np.float32).reshape((-1,5))
        else:
            pts_prev_path = tk_pts[img_metas[0]['prev_idx']]
            pts_lidar_prev = np.fromfile(pts_prev_path, dtype=np.float32).reshape((-1,5))
        pts_lidar_curt = np.fromfile(pts_curt_path, dtype=np.float32).reshape((-1,5))
        
        # 0. 分别读入数据，并生成dnst，并做filt
        dnst_grid_prev = gpg.density_grid_gen(pts_lidar_prev)
        dnst_grid_curt = gpg.density_grid_gen(pts_lidar_curt)
         
        # 高度+高差过滤(最终采用)
        dnst_grid_prev = gpg.num_hdt_filt(dnst_grid_prev, 0.5, 2)
        dnst_grid_curt = gpg.num_hdt_filt(dnst_grid_curt, 0.5, 2)
        
        # 同时生成当前帧，补2d前一帧
        # 1. 分别获取欧式空间下的3个参考点(3d_curt,2d_curt,2d_prev) (当prev=none时prev=curt)
        guide_3d, guide_2d_curt = gpg.guide_gen_curt(dnst_grid_curt)
        guide_2d_prev = gpg.guide_gen_2d(dnst_grid_prev)
        
        return guide_2d_prev, guide_2d_curt, guide_3d  # 3d是几何值 2d是ratio

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                pts_feats,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []
        # (50,50,4)   shape=[1,4,22500,3]
        ref_3d = self.get_reference_points(bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, 
                                           dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        # (50,50)  shape=[1,22500,1,2]
        ref_2d = self.get_reference_points(bev_h, bev_w, 
                                           dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        
        # 现有的: 
        # 1. 非欧式或特征空间下的3d点
        # 2. 非欧式或特征空间下的2d点
        
        # TODO:
        guide_2d_prev, guide_2d_curt, guide_3d_curt  = self.guide_obtain(kwargs['img_metas'])
        
        # 2. 3d_curt在point_sampling中与ref_3d融合 (guide_3d新增多少点，就在ref_3d中减去多少点，保证shape=[22500,4])
        # 3. 2d_prev和2d_curt转换到特征空间下
        
        # BEV空间感知范围：前后左右51.2m，上3m，下-5m
        # 获取ref_3d在img上的投影点(lidar2cam)
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, guide_3d_curt, kwargs['img_metas'])
        
        # guide_2d_prev, guide_2d_curt = space_trans(guide_2d_prev, guide_2d_curt)
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]  # 通过can_bus算出来的帧间bev_grid偏移量
        
        # 4. 将转换为特征空间后的guide_2d与ref_2d分别融合，然后输出2d_hybird 

        num_2d_curt = guide_2d_curt.shape[0]
        guide_2d_curt = torch.from_numpy(guide_2d_curt)
        ref_2d[0,:num_2d_curt,0,:]=guide_2d_curt
        num_2d_prev = guide_2d_prev.shape[0]
        guide_2d_prev = torch.from_numpy(guide_2d_prev)
        shift_ref_2d[0,:num_2d_prev,0,:]=guide_2d_prev

        # 想要的: 
        # 1. ref_3d with 3d guide shape=22500 
        # 2. 3d经ref_cam后，已经变成了相对于图片尺寸的ratio
        # 3. hybird_ref_2d 分别由curt和prev组成
        # 4. 2d中包含 2d guide shape=22500，缩放成特征空间尺寸了

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            # prev_bev与bev_query融合
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            # 平移前后的ref_2d点融合
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
            
        # 6轮bevformer_layer
        # print('\nencoder:')
        # exchange = 1024**3
        # a = (torch.cuda.memory_allocated())/exchange
        # b = (torch.cuda.memory_reserved())/exchange
        # print("  allocated:{:.2f}GB".format(a))
        # print("  reserved:{:.2f}GB".format(b))
        for lid, layer in enumerate(self.layers):
            # print(f'    round-{lid}:')
            # exchange = 1024**3
            # c = torch.cuda.memory_allocated()/exchange
            # d = torch.cuda.memory_reserved()/exchange
            # print("      allocated:{:.2f}GB".format(c))
            # print("      reserved:{:.2f}GB".format(d))
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                pts_feats = pts_feats,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 pts_fusion_layer=None,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 9
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'norm',
              'senet_fusion','semi_attn', 'norm', 'ffn', 'norm'])
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(pts_fusion_layer)

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                pts_feats = None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'
                
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                # print(f"layer: {layer}")
                query = self.attentions[attn_index](  # 0:TSA  1:SCA  所以2可以是semi_fusion_atten
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                # print(f"layer: {layer}")
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                # print(f"layer: {layer}")
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query
                
            elif layer == 'senet_fusion':     
                bs = 1 # mlvl_feats[0].size(0)
                # img_bev_feature shape: (bs, bev_h*bev_w, embed_dims)即(1, 150x150, 256)
                # pts_feats shape: (1, 384, 128, 128)
                features = []
                # 点云BEV空间特征的尺寸为BEV_H_align, BEV_W_align（128x128）
                # BEV_H_align, BEV_W_align = pts_feats[0].shape[-2:]
                
                # 1. img_bev_feature由(bs, bev_h*bev_w, embed_dims)变换维度到 img_bev_feature_align：(bs, self.embed_dims, bev_h, bev_w)
                query_pre_fusion = query.clone().detach()
                query_pre_fusion = query_pre_fusion.reshape(bs, bev_h, bev_w, self.embed_dims).permute(0,3,1,2)
                
                # 2. 将query特征通过interpolate插值函数插值到128x128大小，与点云BEV对齐，mode选择bilinear双线性插值法，设置对齐corners  scale_factor=(2.56,2.56)
                # query = F.interpolate(query, (BEV_H_align, BEV_W_align), mode='bilinear', align_corners=True)
                
                # 2. 将pts_feats特征通过interpolate插值函数插值到150x150大小，与query对齐，mode选择bilinear双线性插值法，设置对齐corners  scale_factor=(2.56,2.56)
                pts_feats[0] = F.interpolate(pts_feats[0], (bev_h, bev_w), mode='bilinear', align_corners=True)
                # 3. 在features列表中添加上图像BEV特征和点云BEV特征
                features.append(query_pre_fusion)
                features.append(pts_feats)
        
                # 4. 将features列表输入给融合层self.pts_fusion_layer做融合加强, [1, 384+256, 150, 150]-->[1, 256, 150, 150]
                fusion_bev = self.pts_fusion_layer(features)
                fusion_bev = fusion_bev.permute(0,2,3,1).reshape(bs, -1, self.embed_dims)
                
            elif layer == 'semi_attn':
                # print(f"layer: {layer}")
                query = self.attentions[attn_index](
                    query,
                    fusion_bev,
                    fusion_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d[1].unsqueeze(0),  # ref_2d = stack(prev,curt) 现在只用取curt的即可(但是要加一维dim=0)
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # print(f"layer: {layer}")
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
            
        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention

@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
