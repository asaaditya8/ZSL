import torch
from torch import nn

import math


class InvertedResidual(nn.Module):
    def __init__(self, planes, out_planes, stride, expansion, dropout=0.3, res_scale=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.res_scale = res_scale
        assert stride in [1, 2]

        h_planes = planes * expansion
        self.use_residual = self.stride == 1 and planes == out_planes

        if expansion == 1:
            self.block = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                          groups=planes),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(planes, h_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(h_planes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(h_planes, h_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                          groups=h_planes),
                nn.BatchNorm2d(h_planes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(h_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)*self.res_scale
        else:
            return self.block(x)


class MobileNet(nn.Module):
    """
    This is invertedresidual resnet, with dropout and res_scale. Add DNI here. Use this for training from scratch.
    Dropout can replace the need for data augmentation.
    """
    def __init__(self, network_settings, num_classes=1000, dropout=0.3, res_scale=1):
        super(MobileNet, self).__init__()
        self.network_settings = network_settings
        self.num_classes = num_classes
        self.dropout = dropout
        self.res_scale = res_scale

        input_channel = network_settings[0]
        last_channel = network_settings[-1]

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )]

        for t, c, n, s in network_settings[1:-1]:
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, c, s, expansion=t, dropout=0.3,  res_scale=self.res_scale))
                else:
                    self.features.append(InvertedResidual(input_channel, c, 1, expansion=t, dropout=0.3,  res_scale=self.res_scale))
                input_channel = c

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(**kwargs):
    """
    2 times the width of ResNet50
    """
    model = MobileNet([
        # t, c, n, s
        32,
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
        1280
    ], **kwargs)
    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 416, 416))
    print(mobilenet_v2().state_dict().keys())