from torch import nn


class Conv(nn.Module):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 idx=None, from_idx=-1, act=default_act):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.act = act()
        self.idx = idx
        self.from_idx = from_idx

    def forward(self, x):
        return self.act(self.conv(x))


class ConvBN(Conv):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 idx=None, from_idx=-1, act=default_act):
        super().__init__(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, idx, from_idx, act)
        self.bn = nn.BatchNorm2d(out_ch)
        self.idx = idx
        self.from_idx = from_idx

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, idx=None, from_idx=-1):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size, stride)
        self.idx = idx
        self.from_idx = from_idx

    def forward(self, x):
        return self.avg(x)


class FC(nn.Module):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, idx=None, from_idx=-1, act=default_act):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.act = act()
        self.idx = idx
        self.from_idx = from_idx

    def forward(self, x):
        return self.act(self.fc(x))


class Softmax(nn.Module):
    def __init__(self, dim=1, idx=None, from_idx=-1):
        super().__init__()
        self.sm = nn.Softmax(dim=dim)
        self.idx = idx
        self.from_idx = from_idx

    def forward(self, x):
        return self.sm(x)
