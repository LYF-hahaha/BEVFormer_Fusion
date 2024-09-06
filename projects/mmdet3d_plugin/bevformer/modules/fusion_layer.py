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
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            SE_Block(self.out_channels)
        )
#TODO 1: replace the 'None' values in the following code with correct expressions.
# input is a list containing two Tensors with different channels, you should 
# concatenate them in channel dimension and then give to the super().forward() as input.
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(
            None
            )

#TODO 2: replace the 'None' values in the following code with correct expressions.
# self.att module consists of AdaptiveAvgPool2d, 1x1convolution and sigmoid function
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            None,
            None,
            None
        )
        
#TODO 3: complete the forward process.
# weights should be distributed for each channel to get final output.
    def forward(self, x):
        
        raise NotImplementedError