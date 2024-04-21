import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock2d
from .layers import ResidualBlock2d
from .layers import NodeAttention


class SpectrogramCnnModel(nn.Module):
    layers = [
        16,
        32,
        48,
        64, 
        128,
        256
    ]
    def __init__(self, in_channels: int = 1):
        super(SpectrogramCnnModel, self).__init__()

        embed_dim = self.layers[-1]

        self.projection = ConvBlock2d(in_channels, self.layers[0])
        self.encoder = nn.Sequential(
            ResidualBlock2d(self.layers[0], self.layers[1]), # (96, 224) -> (48, 112)
            ResidualBlock2d(self.layers[1], self.layers[2]), # (48, 112) -> (24,  56) 
            ResidualBlock2d(self.layers[2], self.layers[3]), # (24,  56) -> (12,  28) 
            ResidualBlock2d(self.layers[3], self.layers[4]), # (12,  28) -> ( 6,  14) 
            ResidualBlock2d(self.layers[4], self.layers[5]), # ( 6,  14) -> ( 3,   7) 
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.node_attention = NodeAttention(input_size = embed_dim, embed_dim = embed_dim, n_nodes = 4)
        self.pool_factor = nn.Linear(embed_dim, 1)

        self.decoder = nn.Linear(embed_dim, 6)
        self.temperature = nn.Sequential(
            nn.Linear(embed_dim, 6),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embedding(x)
        x = self.forward_head(x)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        b = x.shape[0]

        # Make a decision based on relationships in each node of the montage. 
        x = self.node_attention(x)
        pf = F.softmax(self.pool_factor(x), dim = 1) 
        x = torch.sum(x * pf, dim = 1).view(b, -1)

        if pre_logits:
            return x

        t = self.temperature(x)
        logits = self.decoder(x)
        pred = F.log_softmax(logits / t, dim = -1)

        return pred

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = x.view(b * c, 1, h, w)

        x = self.projection(x) # (b * c,   1, h  , w  )
        x = self.encoder(x)    # (b * c, 128, h_n, w_n)
        x = self.pool(x)       # (b * c, 128)
        x = x.view(b, c, -1)   # (b,  c, 128)

        return x


