import torch
from torch import nn

from .CA import CA_base


class FeatureAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, dim); reweight features per sample
        weights = torch.softmax(self.fc(x), dim=1)
        return x * weights


class CA0Att(CA_base):
    """
    Same interface as CA0 but inserts a feature-wise attention layer before the linear projection
    for both beta and factor branches.
    """

    def __init__(self, hidden_size, lr=0.001, omit_char=None, device="cuda", l1_lambda=None, l1_candidates=None, hidden_width=64):
        if omit_char is None:
            omit_char = []
        super().__init__(name=f"CA0Att_{hidden_size}", omit_char=omit_char, device=device, l1_lambda=l1_lambda, l1_candidates=l1_candidates)

        self.beta_nn = nn.Sequential(
            FeatureAttention(94),
            nn.Linear(94, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_size),
        )
        self.factor_nn = nn.Sequential(
            FeatureAttention(94),
            nn.Linear(94, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_size),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
