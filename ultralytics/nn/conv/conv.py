import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
#__all__ = ['SPDConv']
 
class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
 
    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

#
class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.contiguous().view(1, -1, h, w)
        weight = self.weight.contiguous().view(k, -1)

        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output

class DynamicConv(nn.Module):
    """动态卷积层，使用条件卷积（CondConv2d）实现。"""
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, num_experts=4):
        """
        初始化动态卷积层。
        参数：
        in_features : 输入特征通道数
        out_features : 输出特征通道数
        kernel_size : 卷积核大小
        stride : 步长
        padding : 填充
        dilation : 膨胀系数
        groups : 组数
        bias : 是否使用偏置
        num_experts : 专家数量（用于CondConv2d）
        """
        super().__init__()
        # 路由层，用于计算每个专家的权重
        self.routing = nn.Linear(in_features, num_experts)
        # 条件卷积层，实现动态卷积
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        """前向传播函数，实现动态路由和条件卷积的应用。"""
        # 先对输入进行全局平均池化，并展平
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # 计算路由权重
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        # 应用条件卷积
        x = self.cond_conv(x, routing_weights)
        return x