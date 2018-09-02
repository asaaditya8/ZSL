import torch
from torch import nn
from torch.nn import functional as F

from MobileNetV2 import MobileNetV2

device = torch.device('cuda')

net = MobileNetV2(include_top=False)
state_dict = torch.load('mobilenet_v2.pth.tar') # add map_location='cpu' if no gpu
net.load_state_dict(state_dict, strict=False)

input_channel = 1280
net.add_module('splitter', nn.Conv2d(input_channel, input_channel, 1, bias=False))


def to_var(x):
    #     if torch.cuda.is_available():
    #         x = x.cuda()
    global device
    return x.to(device)


def idx2onehot(idx, n):
    assert idx.size(1) == 1
    assert torch.max(idx).data[0] < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)

    return onehot


def detection(ip_dim):
    # current size of feature is 1280x13x13
    return nn.Conv2d(ip_dim, 20, 1)


def classifier(ip_dim):
    # current size of feature is 1280x13x13
    # number of class in coco = 65
    return nn.Conv2d(ip_dim, 65, 1)


def encoder_stem(ip_dim, hidden_dim):
    return nn.Sequential(
        nn.Conv2d(ip_dim, hidden_dim, 1),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.ELU(inplace=True),
    )


def decoder_stem(att_dim, latent_dim, hidden_dim, out_dim):
    ip_dim = att_dim + latent_dim
    return nn.Sequential(
        nn.Conv2d(ip_dim, hidden_dim, 1),
        nn.ELU(),
        nn.Conv2d(hidden_dim, out_dim, 1)
    )


def regressor_stem(ip_dim, att_dim, hidden_dim):
    return nn.Sequential(
        nn.Conv2d(ip_dim, hidden_dim, 1),
        nn.ELU(),
        nn.Conv2d(hidden_dim, att_dim, 1)
    )


class Encoder(nn.Module):

    def __init__(self, ip_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.conv = encoder_stem(ip_dim, hidden_dim)
        self.conv_means = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.conv_var = nn.Conv2d(hidden_dim, latent_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        means = self.conv_means(x)
        log_vars = torch.log(F.softplus(self.conv_var))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_dim, att_dim, hidden_dim, out_dim, embeddings):
        super(Decoder, self).__init__()

        self.embeddings = embeddings
        self.conv = decoder_stem(att_dim, latent_dim, hidden_dim, out_dim)

    def forward(self, z, c):
        shape = c.size()
        # correction needed
        a = self.embeddings(c.view(-1)).view(shape)
        z = torch.cat((z, a), dim=-1)
        x = self.conv(z)
        return x


class Regressor(nn.Module):

    def __init__(self, layer_sizes, attributes, ip_dim, hidden_dim):
        super().__init__()

        self.num_labels = attributes.shape[0]
        self.att_dim = attributes.shape[1]
        self.attributes = torch.Tensor(attributes, device=device)
        self.attributes = self.attributes.view((1, 1, self.num_labels, self.attribute_size))

        self.conv = regressor_stem(ip_dim, self.att_dim, hidden_dim)

    def forward(self, x):
        # batch_size x 256 x 13 x 13
        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        grid_size = a.size(1)
        # Reshape predicted attribute because broadcasting is not supported
        # a has shape               batch_size x 169 x   1 x 300
        # attributes have shape:             1 x   1 x  65 x 300
        a = a.view((-1, grid_size**2, 1, self.att_dim))
        # logits of shape:            batch_size x 169 x 65
        logits = nn.CosineSimilarity(dim=2, eps=1e-6)(a, self.attributes)
        # log of probabilities shape: batch_size x 169 x 65
        c_hat = F.log_softmax(logits, dim=-1)
        return c_hat


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        shape = mu.size()
        eps = torch.randn(shape, device=device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, shape):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(shape, device=device)
        return z

    def sample_c_prior(self, shape):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = torch.randint(0, self.num_labels, shape, device=device)
        return c

    def forward(self, x):
        """
        Params:
        -------
        c: whether to sample `c` from prior or use what is provided.
        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """

