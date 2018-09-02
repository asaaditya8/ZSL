import torch
import torch.nn.functional as F
from torch import nn

from scratch_models.abstract_network import Operation, Reduction
from functools import partial


class ResNet(nn.Module):
    """
    This is bottleneck resnet.
    """
    def __init__(self, layer_sizes, n_blocks, num_classes=1000, stride=2, dropout=0.3, expansion=4, res_scale=1):
        super(ResNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_blocks = n_blocks
        self.num_classes = num_classes
        self.stride = stride
        self.dropout = dropout
        self.expansion = expansion
        self.res_scale = res_scale
        self.shortcut_fn = lambda x, h: F.relu(x + h * self.res_scale)

        self.conv1 = nn.Conv2d(3, self.layer_sizes[0]*self.expansion, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.layer_sizes[0]*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ops = nn.ModuleList([Operation(partial(self.bottleneck, expansion=self.expansion, dropout=self.dropout), n, i, i,
                                            shortcut_fn=self.shortcut_fn)
                                  for n, i in zip(self.n_blocks, self.layer_sizes[1:])])

        self.reds = nn.ModuleList([Reduction(partial(self.downsample, stride=self.stride, expansion=self.expansion, dropout=self.dropout), i, o,
                                             shortcut_fn=partial(self.downshortcut, stride=self.stride, expansion=self.expansion),
                                             combination_fn=self.shortcut_fn)
                                   for i, o in zip(self.layer_sizes[1:], self.layer_sizes[2:])])

        self.fc = nn.Linear(layer_sizes[-1] * self.expansion, num_classes)

    @staticmethod
    def bottleneck(planes, _, expansion, dropout):
        return [
            nn.Conv2d(planes * expansion, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * expansion)
        ]

    @staticmethod
    def downshortcut(in_factor, out_factor, stride, expansion):
        return [
            nn.Conv2d(in_factor * expansion, out_factor * expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_factor * expansion)
        ]

    @staticmethod
    def downsample(in_factor, out_factor, stride, expansion, dropout):
        inplanes = in_factor * expansion
        return [
            nn.Conv2d(inplanes, in_factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_factor),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_factor, in_factor, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_factor),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_factor, out_factor * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_factor * expansion)
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for op, red in zip(self.ops, self.reds):
            x = op(x)
            x = red(x)

        x = x.mean(3).mean(2)
        x = self.fc(x)
        return x


def resnet50(dropout=0):
    model = ResNet([64, 64, 128, 256, 512], [3, 4, 6, 3], dropout=dropout)
    return model


def wideresnet50_2(**kwargs):
    """
    2 times the width of ResNet50
    """
    model = ResNet([128, 128, 256, 512, 1024], [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 416, 416))
    print(wideresnet50_2()(x).size())