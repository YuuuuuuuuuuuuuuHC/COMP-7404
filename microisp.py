import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None, bn=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.activation = activation
        self.activation_func = None
        if self.activation == "tanh":
            self.activation_func = nn.Tanh()
        if self.activation == "relu":
            self.activation_func = nn.PReLU()
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation == "sigmoid": 
            out = torch.sigmoid(out)
        else:
            out = self.activation_func(out)
        return out

class ChannelAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1, pool_type='avg'):
        super(ChannelAtt, self).__init__()
        self.conv1 = ConvLayer(4, 4, kernel_size=1, stride=3, activation='relu', bn=False)
        self.conv2 = ConvLayer(4, 4, kernel_size=3, stride=3, activation='relu', bn=False)
        self.conv3 = ConvLayer(4, 4, kernel_size=3, stride=3, activation='relu', bn=False)
        self.conv4 = ConvLayer(4, 4, kernel_size=3, stride=3, activation='relu', bn=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_type = pool_type

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.pool_type == 'avg':
            avg_pool = self.avg_pool(x)
            channel_att_raw = self.mlp(avg_pool)
        channel_att_sum = channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale

class ResAtt(nn.Module):
    def __init__(self):
        super(ResAtt, self).__init__()
        self.conv1 = ConvLayer(4, 4, kernel_size=3, stride=1, activation='relu', bn=False)
        self.conv2 = ConvLayer(4, 4, kernel_size=3, stride=1, activation='relu', bn=False)
        self.conv3 = ConvLayer(4, 4, kernel_size=3, stride=1, activation='relu', bn=False)
        self.ChannelAtt = ChannelAtt(gate_channels=4, reduction_ratio=2, pool_type='avg')
        
    def forward(self, x):
        # x [1, 4, 128, 128]
        out1 = self.conv1(x)
        out = self.conv2(out1)
        out = self.conv3(out)
        scale = self.ChannelAtt(out)
        scale = scale.expand_as(out)
        out = out * scale + out1
        return out

class ISPBlock(nn.Module):
    def __init__(self):
        super(ISPBlock, self).__init__()
        self.ResAtt1 = ResAtt()
        self.ResAtt2 = ResAtt()
        self.conv3R = ConvLayer(4, 4, kernel_size=3, stride=1, activation='relu', bn=False)
        self.conv3T = ConvLayer(4, 4, kernel_size=3, stride=1, activation='tanh', bn=False)

    def forward(self, x):
        out = self.ResAtt1(x)
        out = self.ResAtt2(out)
        out = self.conv3R(out)
        out = self.conv3T(out)
        return out

class MicroISPNet(nn.Module):
    def __init__(self, block_num=2):
        super(MicroISPNet, self).__init__()
        self.block_num = block_num
        self.ISPBlockR = ISPBlock()
        self.ISPBlockG = ISPBlock()
        self.ISPBlockB = ISPBlock()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        xR = x
        xG = x
        xB = x

        for i in range(self.block_num):
            xR = self.ISPBlockR(xR)
            xG = self.ISPBlockG(xG)
            xB = self.ISPBlockB(xB)

        xR = self.ps(xR)
        xG = self.ps(xG)
        xB = self.ps(xB)

        res = torch.cat((xR, xG, xB), dim=1)
        return res

if __name__ == '__main__':
    a = np.ones([1, 4, 224, 224])
    a = torch.from_numpy(a).float()
    # testnet = ConvLayer(4,4,3,1,activation='sigmoid',bn=True)
    # testnet = ChannelAtt(gate_channels=4,reduction_ratio=1,pool_type='avg')
    testnet = MicroISPNet()
    b = testnet(a)
    print(b.shape)

