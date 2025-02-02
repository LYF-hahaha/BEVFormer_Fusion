import torch.nn as nn
import torch
from typing import List

# 增加FUSION_LAYERS注册
from mmdet3d.models.builder import FUSION_LAYERS

# register the fusion_layer
@FUSION_LAYERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            SE_Block(self.out_channels)
        )
#TODO 1: replace the 'None' values in the following code with correct expressions.
# input is a list containing two Tensors with different channels, you should 
# concatenate them in channel dimension and then give to the super().forward() as input.
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:    # 类型提示，告知期待的输入和输出类型
        feat_1, feat_2 = inputs
        feat_2 = feat_2[0]
        feat_comb = torch.cat([feat_1, feat_2], dim=1)   # dim瞎写的,好像蒙对了...   更正: 在通道维度cat，feat_1.shape=[1,256,128,128]  fefat_2.shape=[1,384,128,128]，所以是dim=1
        return super().forward(feat_comb)                # dim=1好像就是channel维度 （自信点，把好像俩字去了）

#TODO 2: replace the 'None' values in the following code with correct expressions.
# self.att module consists of AdaptiveAvgPool2d, 1x1convolution and sigmoid function
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # 指定输出尺寸为1×1 (squeeze操作，求和为shape=[1,1])
            nn.Conv2d(256, 256,           # channel wise fusion
                      kernel_size=(1,1)),
            nn.Sigmoid()
        )
        
#TODO 3: complete the forward process.
# weights should be distributed for each channel to get final output.
    def forward(self, x):   # x.shape=[1,256,128,128]
        out = self.att(x)   # (squeeze + excitation，求和为shape=[1,1])
        out = out*x    # scale
        return out
        # raise NotImplementedError
        
        
# def fusion_opreation():
#         img_bev_feat_orig = img_bev_feature.clone().detach()
        
#         bs = mlvl_feats[0].size(0)

#         # img_bev_feature shape: (bs, bev_h*bev_w, embed_dims)即(1, 50x50, 256)
#         # pts_feats shape: (1, 384, 128, 128)

#         features = []
#         # 点云BEV空间特征的尺寸为BEV_H_align, BEV_W_align（128x128）
#         # BEV_H_align, BEV_W_align = pts_feats.shape[-2:]
#         BEV_H_align, BEV_W_align = pts_feats[0].shape[-2:]

#         # TODO: complete the following code with correct expressions.
#         # 1. img_bev_feature由(bs, bev_h*bev_w, embed_dims)变换维度到 img_bev_feature_align：(bs, self.embed_dims, bev_h, bev_w)
#         img_bev_feature = img_bev_feature.reshape(bs, bev_h, bev_w, self.embed_dims).permute(0,3,1,2)
        
#         # 2. 将图像BEV特征通过interpolate插值函数插值到128x128大小，与点云BEV对齐，mode选择bilinear双线性插值法，设置对齐corners  scale_factor=(2.56,2.56)
#         img_bev_feature = F.interpolate(img_bev_feature, (BEV_H_align, BEV_W_align), mode='bilinear', align_corners=True)
        
#         # 3. 在features列表中添加上图像BEV特征和点云BEV特征
#         features.append(img_bev_feature)
#         features.append(pts_feats)
        
#         # 4. 将features列表输入给融合层self.pts_fusion_layer做融合加强, [1, 384+256, 128, 128]-->[1, 256, 128, 128]
#         bev_embed = self.pts_fusion_layer(features)
        
#         # 5. 融合后的BEV特征bev_embed还原到初始形状, 
#         # [bs, self.embed_dims, BEV_H_align, BEV_W_align]-->[bs, BEV_H_align*BEV_W_align, self.embed_dims]
#         bev_embed = bev_embed.reshape(bs,self.embed_dims, BEV_H_align*BEV_W_align).permute(0,2,1)
        
#         # object_query_embed在这里被分解成了query_pos和query
#         query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
#         query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
#         query = query.unsqueeze(0).expand(bs, -1, -1)
        
#         reference_points = self.reference_points(query_pos)   # nn.Linear
#         reference_points = reference_points.sigmoid()
#         init_reference_out = reference_points

#         query = query.permute(1, 0, 2)
#         query_pos = query_pos.permute(1, 0, 2)
#         bev_embed = bev_embed.permute(1, 0, 2)