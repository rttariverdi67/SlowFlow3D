import torch
from networks.flownet3d.util import make_mlp
from networks.flownet3d.layers import SetConvLayer, FlowEmbeddingLayer
from typing import Tuple


class PointMixtureNet(torch.nn.Module):
    """
    PointFeatureNet which is the first part of FlowNet3D and consists of one FlowEmbeddingLayer

    References
    ----------
    .. FlowNet3D: Learning Scene Flow in 3D Point Clouds: Xingyu Liu, Charles R. Qi, Leonidas J. Guibas
       https://arxiv.org/pdf/1806.01411.pdf
    """
    def __init__(self):
        super(PointMixtureNet, self).__init__()
        fe_mlp_1 = make_mlp(128+3, [128, 128, 128])
        self.fe_1 = FlowEmbeddingLayer(r=5.0, sample_rate=1.0, mlp=fe_mlp_1)

        set_conv_mlp_1 = make_mlp(128+3, [128, 128, 256])
        self.set_conv_1 = SetConvLayer(r=2.0, sample_rate=0.25, mlp=set_conv_mlp_1)

        set_conv_mlp_2 = make_mlp(256+3, [256, 256, 512])
        self.set_conv_2 = SetConvLayer(r=4.0, sample_rate=0.25, mlp=set_conv_mlp_2)

    def forward(self, x1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                x2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.tensor:
        """
        """
        fe = self.fe_1(x1, x2)
        x = self.set_conv_1(fe)
        x = self.set_conv_2(x)

        print("-"*100)
        features, pos, batch = x
        print("Output after PointMixture")
        print(features.shape)
        print(pos.shape)
        print(batch.shape)

        return fe, x