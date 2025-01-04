import torch.nn as nn
import torch
# 增加FUSION_LAYERS注册
from mmdet3d.models.builder import FUSION_LAYERS
from typing import List

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