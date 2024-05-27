import torch
from torch import nn
from collections import OrderedDict


class Myconv(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, padding, activate=None, is_bn=True):
        super(Myconv, self).__init__()

        self.conv = nn.Sequential()

        self.conv.add_module("conv", nn.Conv2d(in_c, out_c, kernel_size, stride, padding))

        if is_bn:
            self.conv.add_module("bn", nn.BatchNorm2d(out_c))

        if activate is not None:
            if activate == "LeakyReLU":
                self.conv.add_module("leaky relu", nn.LeakyReLU(0.2, True))
            elif activate == "Sigmoid":
                self.conv.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class DiscriminatorPixelGAN(nn.Module):

    def __init__(self, in_c, out_c, ndf):
        super(DiscriminatorPixelGAN, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ("conv 1", Myconv(in_c+out_c, ndf*2, (3,3), 1, 0, activate="LeakyReLU", is_bn=False)),
            ("conv 2", Myconv(ndf*2, ndf, 1, 1, 0, activate="LeakyReLU", is_bn=True)),
            ("conv 3", Myconv(ndf, 1, 1, 1, 0, activate="Sigmoid", is_bn=False))
        ]))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, in_c, out_c, ndf, n_layers=0):
        super(Discriminator, self).__init__()

        """
        n=0 -> rf = 1
        n=1 -> rf = 16
        n=2 -> rf = 34
        n=3 -> rf = 70
        n=4 -> rf = 142
        n=5 -> rf = 286
        n=6 -> rf = 574
        """
        if n_layers == 0:
            self.discrininator = DiscriminatorPixelGAN(in_c, out_c, ndf)
        else:
            # PatchGAN
            self.discrininator = nn.Sequential()

            self.discrininator.add_module("conv input", Myconv(in_c+out_c, ndf, 3, 2, 1,
                                                               activate="LeakyReLU", is_bn=False)),

            nf_mult = 1
            nf_mult_prev = 1
            for i in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**i, 8)
                self.discrininator.add_module("conv {}".format(i), Myconv(ndf*nf_mult_prev, ndf*nf_mult, 3, 2, 1,
                                                                          activate="LeakyReLU", is_bn=True))

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)

            self.discrininator.add_module("conv {}".format(n_layers), Myconv(ndf*nf_mult_prev, ndf*nf_mult, 3, 1, 1,
                                                                             activate="LeakyReLU", is_bn=True))

            self.discrininator.add_module("conv output", Myconv(ndf*nf_mult, 1, 3, 1, 1,
                                                                activate="Sigmoid", is_bn=False))

    def forward(self, x1, x2):
        input_tensor = torch.cat([x1, x2], dim=1)
        return self.discrininator(input_tensor)

    def weights_init(self, mod):
        """设计初始化函数"""
        classname = mod.__class__.__name__
        if classname.find('Conv') != -1:  # 这里的Conv和BatchNnorm是torc.nn里的形式
            mod.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            mod.weight.data.normal_(1.0, 0.02)  # bn层里初始化γ，服从（1，0.02）的正态分布
            mod.bias.data.fill_(0)  # bn层里初始化β，默认为0


if __name__ == "__main__":

    in_c = 1
    out_c = 1
    ndf = 1
    generator_input = torch.randn((10, in_c, 16, 500))
    generator_output = torch.randn((10, out_c, 16, 500))

    discriminator = Discriminator(in_c, out_c, ndf, 3)

    output = discriminator(generator_input, generator_output)
    print(output.shape)
    print()
    # model = DiscriminatorPixelGAN(in_c=3, out_c=3, ndf=1)
    # for name, param in model.named_parameters():
    #     if 'weight' in name:  # 只查看权重参数
    #         print(f"Layer {name}: {param.size()}")
