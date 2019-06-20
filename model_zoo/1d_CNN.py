import torch as t
import torch.nn as nn


class BaseMode(nn.Module):
    def __init__(self,
                 in_channels,
                 n_class):

        super(BaseMode, self).__init__()
        self._make_stem_layer(in_channels)

        """配置block"""

        self.classifilier = nn.ModuleDict()
        self.classifilier.update({
            'fc1': nn.Conv2d(256, 256, 1),
            'relu1': nn.ReLU(),
            'fc2': nn.Conv2d(256, n_class, 1)
        })

    def _make_stem_layer(self, in_channels):
        self.blocks0 = nn.ModuleList()
        self.blocks0.append(nn.Conv2d(in_channels, 64, kernel_size=(7, 1), bias=False))
        self.blocks0.append(nn.BatchNorm2d(num_features=64))
        self.blocks0.append(nn.ReLU(inplace=True))

    def forward(self, input):
        x = input
        for i, b in enumerate(self.blocks):
            x = self.blocks[b](x)
            # x = x.reshape(-1, self.groups[i], self.out_feats[i]//self.groups[i], 1, 1)
            # x = x.reshape(-1, self.out_feats[i] // self.groups[i], self.groups[i], 1, 1)
            # x = x.reshape(-1, self.out_feats[i], 1, 1)

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

