from torch import nn

"""
Encoder
Operation - Repeating blocks
Reduction

class CompositeModel(nn.Module):

    def __init__(self, layer_sizes, n_blocks, stride):
        super(CompositeModel, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_blocks = n_blocks
        self.shortcut_fn = None
        self.stride = stride

        self.ops = [Operation(self.bottleneck, n, i, o, shortcut_fn=self.shortcut_fn)
                    for n, i, o in zip(self.n_blocks, self.layer_sizes[:-1], self.layer_sizes[1:])]

        self.reds = [Reduction(self.bottleneck, i, i, shortcut_fn=partial(self.downsample, stride=self.stride),
                               combination_fn=self.shortcut_fn)
                     for i in self.layer_sizes[1:]]

    def downsample(self, inplanes, planes, stride):
        pass

    def bottleneck(self, inplanes, planes):
        pass
"""


class Operation(nn.Module):
    """
    Takes a function that takes input dim and returns a sequence of layers
    This sequence is repeated using shortcut function
    """
    # And the output is not stored for future use

    def __init__(self, block_fn, n_block, in_dim, out_dim, shortcut_fn=None):
        super(Operation, self).__init__()
        self.block = nn.Sequential(*block_fn(in_dim, out_dim))
        self.n_block = n_block
        self.shortcut_fn = shortcut_fn

    def forward(self, x):
        if self.shortcut_fn is not None:
            output = self.repeat_fn(x)
        else:
            for i in range(self.n_block): x = self.block(x)
            output = x

        return output

    def repeat_fn(self, x):
        for i in range(self.n_block):
            h = self.block(x)
            x = self.shortcut_fn(x, h)
        return x


class Reduction(nn.Module):
    """
    Takes two functions that takes input dim and returns a sequence of layers
    One is used for reduction
    Other is used as a shortcut and can be added or concatenated
    """
    def __init__(self, block_fn, in_dim, out_dim, shortcut_fn=None, combination_fn=None):
        super(Reduction, self).__init__()
        self.block = nn.Sequential(*block_fn(in_dim, out_dim))
        self.shortcut_fn = nn.Sequential(*shortcut_fn(in_dim, out_dim))
        self.combination_fn = combination_fn

    def forward(self, x):
        if self.shortcut_fn is not None:
            h = self.block(x)
            sh = self.shortcut_fn(x)
            output = self.combination_fn(h, sh)
        else:
            output = self.block(x)

        return output

"""
Betwen two pooling steps whatever is there we can categorise that into operation block and reduction block
And save pre and post reduction results. So that they can be retrieved whenever needed.
Incorporate DNI
Make a general class
Then use it to make resnet and mobilenet.
"""