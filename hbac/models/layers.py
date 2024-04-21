import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock1d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = "same", **kwargs) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias = False, **kwargs),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace = True)
    )


def ConvBlock2d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = "same", **kwargs) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )


class ResidualBlock1d(nn.Module):
    """Residual encoder block."""
    def __init__(self, in_channels: int, feature_maps: int, kernel_size: int = 3, padding: int = 1, bias: bool = False, **kwargs) -> None:
        super(ResidualBlock1d, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, ceil_mode = True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, feature_maps, kernel_size = 1, bias = False),
            nn.BatchNorm1d(feature_maps)
        )

        self.dropout = nn.Dropout1d(p = 0.4)

        self.conv1 = nn.Conv1d(in_channels, feature_maps, kernel_size, padding = padding, bias = bias, **kwargs)
        self.bn1 = nn.BatchNorm1d(feature_maps)

        self.conv2 = nn.Conv1d(feature_maps, feature_maps, kernel_size, padding = padding, bias = bias, **kwargs)
        self.bn2 = nn.BatchNorm1d(feature_maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.downsample(identity)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class ResidualBlock2d(nn.Module):
    """Residual encoder block."""
    def __init__(self, in_channels: int, feature_maps: int, kernel_size: int = 3, padding: int = 1, bias: bool = False, **kwargs) -> None:
        super(ResidualBlock2d, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, ceil_mode = True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size = 1, bias = False),
            nn.BatchNorm2d(feature_maps)
        )

        self.dropout = nn.Dropout2d(p = 0.4)

        self.conv1 = nn.Conv2d(in_channels, feature_maps, kernel_size, padding = padding, bias = bias, **kwargs)
        self.bn1 = nn.BatchNorm2d(feature_maps)

        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size, padding = padding, bias = bias, **kwargs)
        self.bn2 = nn.BatchNorm2d(feature_maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.downsample(identity)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        """LayerNorm with an optional bias.

        Args:
            ndim (int): The number of dimensions to use for the layer norm. 
            bias (bool): Whether to include the optional bias term.
            eps (float): Small value used for numerical stability.
        
        """
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class LearnablePosEmbeddings(nn.Module):
    def __init__(self, n_nodes: int = 16, embed_dim: int = 32) -> None:
        super(LearnablePosEmbeddings, self).__init__()

        self.embedding = nn.Embedding(n_nodes, embed_dim)
        self.pos = nn.Parameter(torch.arange(n_nodes).unsqueeze(0), requires_grad = False)
        
    def forward(self) -> torch.Tensor:
        return self.embedding(self.pos)


class AttentionBlock(nn.Module):
    def __init__(self, input_size: int, embed_dim: int = 128, num_heads: int = 2, n_nodes: int = 19) -> None:
        super(AttentionBlock, self).__init__()

        self.projection = nn.Linear(input_size, embed_dim)
        self.pos_embeddings = LearnablePosEmbeddings(n_nodes, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.ln_0 = LayerNorm(embed_dim)
        self.ln_1 = LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b, l, e)

        """
        x = self.projection(x) 
        x = x + self.pos_embeddings()

        x = self.ln_0(x)
        x = x + self.attention(x, x, x, need_weights = False)[0]

        x = self.ln_1(x)
        x = x + self.fc(x)

        return x


class NodeAttention(nn.Module):
    def __init__(self, input_size: int, embed_dim: int, num_layers: int = 2, num_heads: int = 2, n_nodes: int = 19) -> None:
        super(NodeAttention, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.attention_layers = nn.ModuleList([
            AttentionBlock(input_size, embed_dim, num_heads, n_nodes) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.attention_layers[i](x)

        return x


class RnnBlock(nn.Module):
    def __init__(self, input_size: int, embed_dim: int = 32, num_layers: int = 1) -> None:
        super(RnnBlock, self).__init__()

        self.rnn = nn.GRU(
            input_size = input_size, 
            hidden_size = embed_dim, 
            num_layers = num_layers, 
            batch_first = True, 
            bidirectional = False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        embed, _ = self.rnn(x)
        embed = embed[:, -1]

        return embed


