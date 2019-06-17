import torch as t
import torch.nn as nn


def group1d(in_channels, out_channels, groups):
    return \
        nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class BaseMode(nn.Module):
    def __init__(self,
                 in_channels,
                 n_class,
                 groups=(16, 64, 32),
                 out_feats=(1024, 512, 256)):

        super(BaseMode, self).__init__()
        self.groups = groups
        self.out_feats = out_feats
        assert len(groups) == len(out_feats), "groups's len must equal to out_feats's len"

        self.g_blocks = nn.ModuleDict({'gconv1': group1d(in_channels, out_feats[0], groups[0])})
        for i in range(len(groups)):
            if i == 0:
                continue
            self.g_blocks.update({
                'gconv{}'.format(i+1): group1d(out_feats[i-1], out_feats[i], groups[i]),  # cin:16/g, ker:8/g
            })

        self.classifilier = nn.ModuleDict()
        self.classifilier.update({
            'fc1': nn.Conv2d(256, 256, 1),
            'relu1': nn.ReLU(),
            'fc2': nn.Conv2d(256, n_class, 1)
        })

    def forward(self, input):
        x = input
        for i, b in enumerate(self.g_blocks):
            x = self.g_blocks[b](x)
            x = x.reshape(-1, self.groups[i], self.out_feats[i]//self.groups[i], 1, 1)
            x = x.reshape(-1, self.out_feats[i] // self.groups[i], self.groups[i], 1, 1)
            x = x.reshape(-1, self.out_feats[i], 1, 1)

        for l in self.classifilier.values():
            x = l(x)
        x.squeeze_()
        return x  # [n_class]

    def get_loss(self, *args, **kwargs):
        """

        :param preds: [n, n_cls]
        :param labs:  [n]
        :return:
        """
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)

    def parameters_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


if __name__ == "__main__":
    model = BaseMode(2000, 43)
    fluxes = t.randn([2, 2000, 1, 1])
    res = model(fluxes)

