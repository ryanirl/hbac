import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock1d
from .layers import NodeAttention
from .layers import RnnBlock


class SignalEncoder50hz(nn.Module):
    layers = [
        16,
        24,
        32,
        48,
        64,
        96,
        256
    ]
    def __init__(self, in_channels: int = 1, scale: int = 1) -> None:
        """ 
        """
        super(SignalEncoder50hz, self).__init__()

        self.projection = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = self.layers[0] * scale, 
            kernel_size = 5,
            padding = 2
        )
        self.downsample = nn.Sequential(
            ResidualBlock1d(self.layers[0] * scale, self.layers[1] * scale, kernel_size = 5, padding = 2), # 50hz -> 25hz
            ResidualBlock1d(self.layers[1] * scale, self.layers[2] * scale, kernel_size = 5, padding = 2), # 25hz ~> 12hz
            ResidualBlock1d(self.layers[2] * scale, self.layers[3] * scale, kernel_size = 3, padding = 1), # 12hz ~> 6hz
            ResidualBlock1d(self.layers[3] * scale, self.layers[4] * scale, kernel_size = 3, padding = 1), #  6hz ~> 3hz
            ResidualBlock1d(self.layers[4] * scale, self.layers[5] * scale, kernel_size = 3, padding = 1)  #  3hz ~> 1hz
        )
        self.temporal_embed = RnnBlock(
            input_size = self.layers[5] * scale, 
            embed_dim = self.layers[6] * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.downsample(x)
        x = self.temporal_embed(x)
        return x


class EegModel(nn.Module):
    def __init__(self, in_channels: int = 1, scale: int = 1) -> None:
        """ 
        """
        super(EegModel, self).__init__()

        self.ekg_encoder = SignalEncoder50hz(in_channels, scale)
        self.encoder = SignalEncoder50hz(in_channels, scale)
        self.signal_dim = self.encoder.layers[-1] * scale
        self.embed_dim = self.signal_dim

        self.node_attention = NodeAttention(
            input_size = self.signal_dim, embed_dim = self.signal_dim
        )
        self.pool_factor = nn.Linear(self.signal_dim, 1)

        self.decoder = nn.Linear(self.embed_dim, 6) 
        self.temperature = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        """
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

        # Sample-dependent adaptive temperature scaling. 
        t = self.temperature(x)
        logits = self.decoder(x)
        pred = F.log_softmax(logits / t, dim = -1)

        return pred

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        b, c, l = x.shape

        ekg = x[:, c-1].unsqueeze(1)
        x   = x[:, :c-1]

        x = x.view(b, (c - 1), -1)
        x = x.reshape(b * (c - 1), 1, -1)

        # Downsample to ~1hz then pass it to the RNN.
        x = self.encoder(x) 
        x = x.view(b, c - 1, -1)
        ekg = self.ekg_encoder(ekg)
        ekg = ekg.view(b, 1, -1)

        x = torch.concat([x, ekg], dim = 1)

        return x


