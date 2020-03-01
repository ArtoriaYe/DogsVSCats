import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class ResidualBlock(nn.Module):
    """
    实现module: Residual Block: 18layers、34layers
    """
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chanel, out_chanel, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.cov1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chanel)
        self.cov2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chanel)
        self.cov3 = nn.Conv2d(out_chanel, out_chanel*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * out_chanel)
        self.relu = nn.ReLU(out_chanel*4)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.cov1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.cov2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.cov3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.in_chanels = 64

        self.cov1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[1], stride=last_stride)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 2, bias=True)

    def _make_layer(self, block, chanels, blocks, stride=1):
        down_sample = None
        if stride == 1 or self.in_chanels != chanels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_chanels, chanels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(chanels*block.expansion)
            )

        layers = list()
        layers.append(block(self.in_chanels, chanels, stride, down_sample))
        self.in_chanels = chanels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_chanels, chanels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cov1(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()