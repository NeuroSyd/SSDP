import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron

from torch.nn.common_types import _size_2_t


class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class LIFWithTiming(LIF):
    """
    在 LIF 的基础上记录每个神经元的首次脉冲时间。
    输入与输出与 LIF 相同，仅在 forward 执行完毕后，类中多出一个
    `self.first_spike_time` 张量。

    假设输入 x 形状为 [T, B, C, H, W]。
    输出脉冲 spike_output 形状相同。
    在维度 [B, C, H, W] 上记录首次发放脉冲的时间步索引 t (0 <= t < T)。
    若从未发放脉冲则为 -1。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 调用父类forward得到脉冲输出
        # spike_output: [T, B, C, H, W]
        spike_output = super().forward(x)
        T, B, C, H, W = spike_output.shape

        # 将spike_output转为bool，以便查找首次发放时间
        first_spike_mask = (spike_output > 0)  # [T,B,C,H,W]

        # 创建一个时间索引张量 [T,B,C,H,W], 每个元素对应时间步t
        # 利用broadcasting产生[0,1,...,T-1] 的时间索引
        time_indices = torch.arange(T, device=x.device, dtype=torch.long).view(T, 1, 1, 1, 1)
        time_indices = time_indices.expand(T, B, C, H, W)  # [T,B,C,H,W]

        # 对于没有发放脉冲的位置，需要一个标记值（例如T）用于后续取最小值时区分
        # 将未发放脉冲位置填充为一个较大值T
        masked_time = torch.where(first_spike_mask, time_indices,
                                  torch.full((1,), fill_value=T, device=x.device, dtype=torch.long))

        # 沿时间维度取最小值，即首次发放的时间步骤
        # min_values: [B,C,H,W]
        min_values, _ = masked_time.min(dim=0)

        # 如果min_values==T，则说明该位置从未发放脉冲，用-1表示
        self.first_spike_time = torch.where(min_values == T, torch.full_like(min_values, -1), min_values)

        return spike_output


class BN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
        return functional.seq_to_ann_forward(x, self.bn)


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(layer.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: _size_2_t = 1,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, bias=bias, padding_mode='zeros',
                         step_mode='m')


class Conv1x1(layer.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: _size_2_t = 1,
            bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode='zeros', step_mode='m')


class Linear(layer.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, step_mode='m')
