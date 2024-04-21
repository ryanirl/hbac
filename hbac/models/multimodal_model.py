import torch
import torch.nn as nn
import torch.nn.functional as F 

from .eeg_model import EegModel
from .spectrogram_model import SpectrogramCnnModel
from .layers import NodeAttention


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, input_size: int = 256, embed_dim: int = 256, freeze_backbone: bool = True) -> None:
        super(MultimodalModel, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_backbone

        self.eeg_model = EegModel()
        self.eeg_spc_model = SpectrogramCnnModel()
        self.spc_model = SpectrogramCnnModel()

        if freeze_backbone:
            freeze(self.eeg_model)
            freeze(self.eeg_spc_model)
            freeze(self.spc_model)

        self.node_attention = NodeAttention(input_size = input_size, embed_dim = embed_dim, n_nodes = 27)
        self.pool_factor = nn.Linear(embed_dim, 1)
        self.temperature = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())
        self.decode = nn.Linear(embed_dim, 6)

    def forward_embedding(self, eeg: torch.Tensor, eeg_spc: torch.Tensor, spc: torch.Tensor) -> torch.Tensor:
        t0 = self.eeg_model.forward_embedding(eeg)         # (b, 19, E)
        t1 = self.eeg_spc_model.forward_embedding(eeg_spc) # (b,  4, E)
        t2 = self.spc_model.forward_embedding(spc)         # (b,  4, E)

        x = torch.concat([t0, t1, t2], dim = 1) # (b, 27, E)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        b = x.size(0)

        # Make a decision based on relationships in each node of the montage. 
        x = self.node_attention(x)
        pf = F.softmax(self.pool_factor(x), dim = 1) 
        x = torch.sum(x * pf, dim = 1).view(b, -1)

        if pre_logits:
            return x

        t = self.temperature(x)
        logits = self.decode(x)
        pred = F.log_softmax(logits / t, dim = -1)

        return pred

    def forward(self, eeg: torch.Tensor, eeg_spc: torch.Tensor, spc: torch.Tensor):
        x = self.forward_embedding(eeg, eeg_spc, spc)
        x = self.forward_head(x)

        return x


class MultimodalPretrainedBackboneEntrypoint(MultimodalModel):
    def __init__(
        self, 
        eeg_ckpt_path: str,
        eeg_spec_ckpt_path: str,
        spec_ckpt_path: str,
        input_size: int = 256, 
        embed_dim: int = 256, 
        freeze_backbone: bool = True
    ) -> None:
        """
        """
        super().__init__(input_size, embed_dim, freeze_backbone)

        self.eeg_model.load_state_dict(torch.load(eeg_ckpt_path)["model_state_dict"])
        self.eeg_spc_model.load_state_dict(torch.load(eeg_spec_ckpt_path)["model_state_dict"])
        self.spc_model.load_state_dict(torch.load(spec_ckpt_path)["model_state_dict"])

        if freeze_backbone:
            freeze(self.eeg_model)
            freeze(self.eeg_spc_model)
            freeze(self.spc_model)


