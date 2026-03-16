import os
from pathlib import Path

import collections
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .CA import CA_base, MAX_EPOCH
from utils import CHARAS_LIST


class SICGraphConv(nn.Module):
    """
    Simple graph convolution: h_i = W_self x_i + W_neigh mean_{j in N(i)} x_j
    Adj is expected to include self-loops (1 on diagonal).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh = torch.matmul(adj, x) / deg
        return self.lin_self(x) + self.lin_neigh(neigh)


class _GraphMonthDataset(Dataset):
    """Wrap month-level graph samples so DataLoader can batch with collate_fn."""

    def __init__(self, months, model_ref):
        self.samples = []
        for m in months:
            item = model_ref._get_item(m)
            if item is not None:
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        permno, chars, factor, labels, adj = self.samples[idx]
        return (
            permno,
            torch.tensor(chars, dtype=torch.float32),
            torch.tensor(factor, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )


class CA0GNN(CA_base):
    """
    CA0-style two-branch model, but beta branch uses a graph convolution over SIC2 edges.
    Nodes: individual firms in each month; edges: same SIC2 code (full clique with self-loops).
    Factor branch stays as the simple linear map used in CA0.
    """

    def __init__(self, hidden_size, lr=0.001, omit_char=None, device="cuda", l1_lambda=None, l1_candidates=None, hidden_width=64, dropout=0.5, datashare_path=None):
        if omit_char is None:
            omit_char = []
        super().__init__(name=f"CA0GNN_{hidden_size}", omit_char=omit_char, device=device, l1_lambda=l1_lambda, l1_candidates=l1_candidates)

        # reload datashare with SIC2 if a GNN-ready file exists / is provided
        candidate_paths = []
        if datashare_path:
            candidate_paths.append(Path(datashare_path))
        candidate_paths.append(Path("data/datashare_re_gnn.pkl"))
        for p in candidate_paths:
            if p.exists():
                tmp = pd.read_pickle(p).astype(np.float64)
                if any(c.lower() == "sic2" for c in tmp.columns):
                    self.datashare_chara = tmp
                    break

        sic_cols = [c for c in self.datashare_chara.columns if c.lower() == "sic2"]
        if not sic_cols:
            raise ValueError("SIC2 column not found in datashare; run data_prepare_gnn.py so datashare_re.pkl includes SIC2.")
        self.sic_col = sic_cols[0]

        self.dropout = dropout
        self.beta_conv = SICGraphConv(94, hidden_width)
        self.beta_act = nn.ReLU()
        self.beta_dropout = nn.Dropout(self.dropout)
        self.beta_out = nn.Linear(hidden_width, hidden_size)

        # Keep factor branch simple (same as CA0 after simplification)
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)

    def reset_weight(self):
        modules = [self.beta_conv, self.beta_out]
        if self.factor_nn is not None:
            modules.append(self.factor_nn)
        for m in modules:
            if m is None:
                continue
            # modules like SICGraphConv implement reset_parameters; fall back to children for Sequential
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            else:
                for layer in m.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
        if self.optimizer is not None:
            self.optimizer.state = collections.defaultdict(dict)

    def _build_adj(self, sic_series: pd.Series) -> np.ndarray:
        sic_vals = sic_series.values
        adj = (sic_vals[:, None] == sic_vals[None, :]).astype(np.float32)
        # ensure self-loops
        np.fill_diagonal(adj, 1.0)
        return adj

    def _get_item(self, month):
        char_df = self.datashare_chara.loc[self.datashare_chara["DATE"] == month]
        ret_df = self.mon_ret.loc[self.mon_ret["date"] == month]
        if char_df.empty or ret_df.empty:
            return None

        merged = char_df.merge(ret_df[["permno", "ret-rf"]], on="permno", how="inner")
        merged = merged.dropna(subset=CHARAS_LIST + [self.sic_col, "ret-rf"])
        if merged.empty:
            return None

        factor_input = self.portfolio_ret.loc[self.portfolio_ret["DATE"] == month][CHARAS_LIST]
        if factor_input.empty:
            return None

        permno = merged["permno"].values
        chars = merged[CHARAS_LIST].values.astype(np.float32)
        labels = merged["ret-rf"].values.astype(np.float32)
        adj = self._build_adj(merged[self.sic_col])
        factor = factor_input.values.astype(np.float32)  # shape (1, 94)
        return permno, chars, factor, labels, adj

    def dataloader(self, period, shuffle=True):
        mon_list = pd.read_pickle("data/mon_list.pkl")
        mon_list = mon_list.loc[(mon_list >= period[0]) & (mon_list <= period[1])]
        dataset = _GraphMonthDataset(mon_list, self)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=lambda x: x)

    def forward(self, char, pfret, adj):
        beta_hidden = self.beta_conv(char, adj)
        beta_hidden = self.beta_act(beta_hidden)
        beta_hidden = self.beta_dropout(beta_hidden)
        beta = self.beta_out(beta_hidden)

        processed_pfret = self.factor_nn(pfret)
        return torch.sum(beta * processed_pfret, dim=1)

    def train_one_epoch(self, loader):
        if len(loader) == 0:
            return np.inf
        epoch_loss = 0.0
        for batch in loader:
            _, beta_nn_input, factor_nn_input, adj, labels = batch[0]
            beta_nn_input = beta_nn_input.to(self.device)
            factor_nn_input = factor_nn_input.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.forward(beta_nn_input, factor_nn_input, adj)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    def valid_one_epoch(self, loader):
        if len(loader) == 0:
            return np.inf
        epoch_loss = 0.0
        for batch in loader:
            _, beta_nn_input, factor_nn_input, adj, labels = batch[0]
            beta_nn_input = beta_nn_input.to(self.device)
            factor_nn_input = factor_nn_input.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

            output = self.forward(beta_nn_input, factor_nn_input, adj)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    def train_model(self):
        if "saved_models" not in os.listdir("./"):
            os.mkdir("saved_models")

        self.train_dataloader = self.dataloader(self.train_period, shuffle=True)
        self.valid_dataloader = self.dataloader(self.valid_period, shuffle=False)
        self.test_dataloader = self.dataloader(self.test_period, shuffle=False)

        if len(self.train_dataloader) == 0 or len(self.valid_dataloader) == 0:
            raise ValueError(
                f"No samples found for train/valid period {self.train_period} / {self.valid_period}. "
                "Check whether datashare_re_gnn.pkl (with SIC2) covers these months."
            )

        min_error = np.inf
        no_update_steps = 0
        valid_loss = []
        train_loss = []

        for i in range(MAX_EPOCH):
            self.train()
            train_error = self.train_one_epoch(self.train_dataloader)
            train_loss.append(train_error)

            self.eval()
            with torch.no_grad():
                valid_error = self.valid_one_epoch(self.valid_dataloader)
            valid_loss.append(valid_error)

            if valid_error < min_error:
                min_error = valid_error
                no_update_steps = 0
                torch.save(self.state_dict(), f"./saved_models/{self.name}.pt")
            else:
                no_update_steps += 1

            if no_update_steps > 2:
                break

            self.load_state_dict(torch.load(f"./saved_models/{self.name}.pt"))
        return train_loss, valid_loss

    def calBeta(self, month, skip_char=None):
        if skip_char is None:
            skip_char = []
        item = self._get_item(month)
        if item is None:
            raise ValueError(f"No data found for month {month}")
        _, chars, _, _, adj = item
        if skip_char:
            char_df = pd.DataFrame(chars, columns=CHARAS_LIST)
            char_df[skip_char] = 0.0
            chars = char_df.values.astype(np.float32)
        chars = torch.tensor(chars, dtype=torch.float32).to(self.device)
        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        beta_hidden = self.beta_conv(chars, adj)
        beta_hidden = self.beta_act(beta_hidden)
        beta_hidden = self.beta_dropout(beta_hidden)
        return self.beta_out(beta_hidden)

    def calFactor(self, month, skip_char=None):
        if skip_char is None:
            skip_char = []
        factor_input = self.portfolio_ret.loc[self.portfolio_ret["DATE"] == month][CHARAS_LIST]
        if factor_input.empty:
            raise ValueError(f"No factor input found for month {month}")
        factor_nn_input = factor_input.copy()
        if skip_char:
            factor_nn_input[skip_char] = 0.0
        factor_nn_input = torch.tensor(factor_nn_input.values, dtype=torch.float32).to(self.device)
        factor_pred = self.factor_nn(factor_nn_input).T
        self.factor_nn_pred.append(factor_pred)
        return factor_pred

    def inference(self, month):
        assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
        mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
        assert mon_beta.shape[1] == mon_factor.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
        return mon_beta @ mon_factor

    def predict(self, month):
        assert month >= self.test_period[0] and month <= self.test_period[1], f"Month error, {month} is not in test period {self.test_period}"
        lag_factor, mon_beta = self.cal_delayed_Factor(month), self.calBeta(month)
        assert mon_beta.shape[1] == lag_factor.shape[0], f"Dimension mismatch between lag_factor: {lag_factor.shape} and mon_beta: {mon_beta.shape}"
        return mon_beta @ lag_factor

    def visualize_graph(self, month, save_dir="graph_vis", max_nodes=200, seed=42):
        """
        Save a graph visualization for a given month using SIC2 clique edges.
        Falls back to adjacency heatmap if networkx is unavailable.
        """
        char_df = self.datashare_chara.loc[self.datashare_chara["DATE"] == month]
        ret_df = self.mon_ret.loc[self.mon_ret["date"] == month]
        if char_df.empty or ret_df.empty:
            raise ValueError(f"No data found for month {month}")

        merged = char_df.merge(ret_df[["permno", "ret-rf"]], on="permno", how="inner")
        merged = merged.dropna(subset=CHARAS_LIST + [self.sic_col, "ret-rf"])
        if merged.empty:
            raise ValueError(f"No merged rows for month {month}")

        sic_series = merged[self.sic_col].reset_index(drop=True)
        adj = self._build_adj(sic_series)

        # optional subsample for readability
        if adj.shape[0] > max_nodes:
            rng = np.random.default_rng(seed)
            keep_idx = np.sort(rng.choice(adj.shape[0], size=max_nodes, replace=False))
            adj = adj[np.ix_(keep_idx, keep_idx)]
            sic_series = sic_series.iloc[keep_idx].reset_index(drop=True)

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{self.name}_{month}.png")

        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.from_numpy_array(adj)
            pos = nx.spring_layout(G, seed=seed)

            # color by SIC2 bucket
            uniq = sic_series.astype(str).unique()
            color_map = {v: i for i, v in enumerate(uniq)}
            node_colors = [color_map[v] for v in sic_series.astype(str)]

            plt.figure(figsize=(6, 6))
            nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, cmap=plt.cm.tab20)
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.25)
            plt.title(f"Month {month} SIC2 graph (n={adj.shape[0]})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
        except ImportError:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.imshow(adj, cmap="Blues")
            plt.title(f"Adjacency (month {month}, n={adj.shape[0]})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()

        return out_path
