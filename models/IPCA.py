import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.append('../')

from utils import *
from .modelBase import modelBase


class IPCA(modelBase):
    def __init__(self, K, omit_char=[]):
        super(IPCA, self).__init__(f'IPCA_{K}')
        self.K = K
        self.omit_char = omit_char
        self.valid_error = []
        self.dtype = torch.float32  # use float32 for faster GPU throughput (with small regularization)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("CUDA device not available; GPU execution required for this version.")

        # ensure deterministic CuBLAS path is enabled when deterministic algorithms are required
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.manual_seed(10)
        torch.use_deterministic_algorithms(True)

        self.gamma = torch.rand((94, self.K), device=self.device, dtype=self.dtype)
        self.__prepare_data()

    def __prepare_data(self):
        # cache tensors on GPU to avoid CPU<->GPU transfers
        p_charas = pd.read_pickle('data/p_charas.pkl')
        portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
        self.mon_list = pd.read_pickle('data/mon_list.pkl')
        self.char_to_idx = {c: i for i, c in enumerate(CHARAS_LIST)}

        self.Z_by_mon = {}
        self.y_by_mon = {}
        for mon in self.mon_list:
            Z_np = p_charas.loc[p_charas.DATE == mon][CHARAS_LIST].values.astype(np.float32)
            y_np = portfolio_ret.loc[portfolio_ret.DATE == mon][CHARAS_LIST].values.astype(np.float32).T
            self.Z_by_mon[mon] = torch.from_numpy(Z_np).to(self.device)
            self.y_by_mon[mon] = torch.from_numpy(y_np).to(self.device)

    def __valid(self):
        MSE_set = []
        for mon in self.mon_list[(self.mon_list >= self.valid_period[0]) & (self.mon_list <= self.valid_period[1])]:
            Z = self.Z_by_mon[mon]  # N * P
            y = self.y_by_mon[mon]  # N * 1
            beta = Z @ self.gamma  # N * K
            f_hat = torch.linalg.pinv(beta.T @ beta) @ (beta.T @ y)  # K * 1
            residual = y - beta @ f_hat
            MSE_set.append(torch.sum(residual ** 2))

        valid_error = torch.stack(MSE_set).sum().item()
        self.valid_error.append(valid_error)
        return valid_error

    def __gamma_iter(self, gamma_old):
        numer = torch.zeros((94 * self.K, 1), device=self.device, dtype=self.dtype)
        denom = torch.zeros((94 * self.K, 94 * self.K), device=self.device, dtype=self.dtype)
        for mon in self.mon_list[(self.mon_list >= self.train_period[0]) & (self.mon_list <= self.train_period[1])]:
            Z = self.Z_by_mon[mon]  # N * P
            y = self.y_by_mon[mon]  # N * 1
            beta = Z @ gamma_old  # N * K
            f_hat = torch.linalg.pinv(beta.T @ beta) @ (beta.T @ y)  # K * 1

            # Using kron identities to avoid large kron tensors:
            # numer: vec(Z.T @ y @ f_hat.T)  -> (P*K, 1)
            zty = Z.T @ y  # P * 1
            numer = numer + torch.matmul(zty, f_hat.T).reshape(-1, 1)

            # denom: kron(f_hat @ f_hat.T, Z.T @ Z) -> (P*K, P*K)
            f_outer = f_hat @ f_hat.T  # K * K
            ztz = Z.T @ Z  # P * P
            denom = denom + torch.kron(f_outer, ztz)

        # small ridge to stabilize pinv under float32
        eps = 1e-5
        denom_reg = denom + eps * torch.eye(denom.shape[0], device=self.device, dtype=self.dtype)
        gamma_new = (torch.linalg.pinv(denom_reg) @ numer).reshape(self.K, 94).T
        return gamma_new

    def train_model(self):
        update_cnt = 0
        min_valid_err = float('inf')
        best_gamma = torch.zeros((94, self.K), device=self.device, dtype=self.dtype)
        while update_cnt < 5:
            self.gamma = self.__gamma_iter(self.gamma)
            valid_error = self.__valid()
            if valid_error < min_valid_err:
                min_valid_err = valid_error
                best_gamma = self.gamma.clone()
                update_cnt = 0
            else:
                update_cnt += 1

        self.gamma = best_gamma

    def inference(self, month):
        if not len(self.omit_char):
            Z = self.Z_by_mon[month]  # N * P
            y = self.y_by_mon[month]  # N * 1
            beta = Z @ self.gamma  # N * K
            f_hat = torch.linalg.pinv(beta.T @ beta) @ (beta.T @ y)  # K * 1
            return (beta @ f_hat).flatten().detach().cpu().numpy()  # N
        else:
            inference_R = []
            Z = self.Z_by_mon[month]
            y = self.y_by_mon[month]

            for char in self.omit_char:
                idx = self.char_to_idx[char]
                Z_input = Z.clone()
                y_input = y.clone()
                Z_input[:, idx] = 0.0
                y_input[idx, 0] = 0.0
                beta = Z_input @ self.gamma
                f_hat = torch.linalg.pinv(beta.T @ beta) @ (beta.T @ y_input)
                inference_R.append((beta @ f_hat).flatten())

            beta_full = Z @ self.gamma
            f_hat_full = torch.linalg.pinv(beta_full.T @ beta_full) @ (beta_full.T @ y)
            inference_R.append((beta_full @ f_hat_full).flatten())

            return torch.stack(inference_R, dim=0).T.detach().cpu().numpy()  # N * m

    def predict(self, month):
        if self.refit_cnt == 0:
            return self.inference(month)

        lag_f_hat = []
        for mon in self.mon_list[(self.mon_list >= 19870101) & (self.mon_list < month)]:
            Z = self.Z_by_mon[mon]  # N * P
            y = self.y_by_mon[mon]  # N * 1
            beta = Z @ self.gamma  # N * K
            f_hat = torch.linalg.pinv(beta.T @ beta) @ (beta.T @ y)  # K * 1
            lag_f_hat.append(f_hat)

        Z = self.Z_by_mon[month]  # N * P
        beta = Z @ self.gamma  # N * K

        avg_lag_f = torch.stack(lag_f_hat, dim=0).mean(dim=0)  # K * 1
        return (beta @ avg_lag_f).flatten().detach().cpu().numpy()
