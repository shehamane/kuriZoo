from torch import nn


class Layer(nn.Module):
    def __init__(self, idx, from_idx, act=None, aggregation=None):
        super().__init__()
        self.idx = idx
        self.from_idx = from_idx


class Conv(Layer):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, /, *,
                 idx, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)

        if act is None:
            self.act = self.default_act()
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.conv(x))


class ConvBN(Conv):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, /, *,
                 idx, from_idx=-1, act=None):
        super().__init__(in_ch, out_ch, kernel_size, stride, padding, dilation, groups,
                         idx=idx, from_idx=from_idx, act=act)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AvgPool(Layer):
    def __init__(self, kernel_size, stride, /, *, idx, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.avg = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        return self.avg(x)


class MaxPool(Layer):
    def __init__(self, kernel_size, stride, /, *, idx, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.mp = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.mp(x)


class FC(Layer):
    default_act = nn.ReLU

    def __init__(self, in_ch, out_ch, /, *, idx=None, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.act(self.fc(x))


class Softmax(Layer):
    def __init__(self, dim=1, /, *, idx=None, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.sm = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.sm(x)

class Dropout(Layer):
    def __init__(self, p=0.5, /, *, idx=None, from_idx=-1, act=None):
        super().__init__(idx, from_idx)
        self.dp = nn.Dropout(p)

    def forward(self, x):
        return self.dp(x)