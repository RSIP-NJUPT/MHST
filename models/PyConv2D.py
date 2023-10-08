import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[4, 4, 4, 4],
                 pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // out_planes_div[2], kernel_size=pyconv_kernels[2],
                            padding=pyconv_kernels[2] // 2, stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes // out_planes_div[3], kernel_size=pyconv_kernels[3],
                            padding=pyconv_kernels[3] // 2, stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[4, 4, 2],
                 pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // out_planes_div[2], kernel_size=pyconv_kernels[2],
                            padding=pyconv_kernels[2] // 2, stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):
    def __init__(self, inplans, planes, out_planes_div=[2, 2],
                 pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // out_planes_div[0], kernel_size=pyconv_kernels[0],
                            padding=pyconv_kernels[0] // 2, stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // out_planes_div[1], kernel_size=pyconv_kernels[1],
                            padding=pyconv_kernels[1] // 2, stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, out_planes_div, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, out_planes_div=out_planes_div, pyconv_kernels=pyconv_kernels,
                       stride=stride, pyconv_groups=pyconv_groups)
