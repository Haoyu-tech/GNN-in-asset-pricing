import torch
from torch import nn

from .CA import CA_base


class MultiHeadFeatureAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.fc = nn.Linear(dim, dim * num_heads)

    def forward(self, x):
        # x: (batch, dim); apply head-wise feature weights then average heads
        B, P = x.shape
        attn_raw = self.fc(x).view(B, self.num_heads, P)              # (B, H, P)
        attn = torch.softmax(attn_raw, dim=2)                         # head-wise softmax over features
        weights = attn.mean(dim=1)                                    # (B, P)
        return x * weights                                            # reweight features


class CA0AttM(CA_base):
    """
    CA0 variant with multi-head feature attention before the linear projection.
    """
    def __init__(self, hidden_size, num_heads=4, lr=0.001, omit_char=None, device='cuda', l1_lambda=None, l1_candidates=None, hidden_width=64):
        if omit_char is None:
            omit_char = []
        super().__init__(name=f'CA0AttM_{hidden_size}', omit_char=omit_char, device=device, l1_lambda=l1_lambda, l1_candidates=l1_candidates)

        att = MultiHeadFeatureAttention(dim=94, num_heads=num_heads)
        self.beta_nn = nn.Sequential(
            att,
            nn.Linear(94, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            att,
            nn.Linear(94, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
