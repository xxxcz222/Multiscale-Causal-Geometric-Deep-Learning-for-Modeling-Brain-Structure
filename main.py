from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


from utils_torch import MI as calculate_MI            
from utils_torch import calculate_conditional_MI      
from transformer import SingleGraphTransformer
from GraphVAE import GraphDecoder


class GraphDecoupledDataset(Dataset):


    def __init__(self, json_path: str | Path):
        with open(json_path, "r") as f:
            self.samples: List[dict] = json.load(f)

    def __len__(self):  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx: int):  # noqa: D401
        s = self.samples[idx]
        age = torch.tensor(s["age"], dtype=torch.float)

        dist_sMRI = torch.tensor(s["dist_sMRI"], dtype=torch.float)
        dist_dMRI = torch.tensor(s["dist_dMRI"], dtype=torch.float)
        sMRI_edge = torch.tensor(s["sMRI_edge_index"], dtype=torch.long)
        dMRI_edge = torch.tensor(s["dMRI_edge_index"], dtype=torch.long)

        return dist_sMRI, dist_dMRI, sMRI_edge, dMRI_edge, age


def safe_conditional_MI(x: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2:  # e.g. batch_size == 1
        return torch.tensor(0.0, device=x.device)
    return calculate_conditional_MI(x, z, y)


def joint_uncond(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    data: Data,
    causal_decoder: GraphDecoder,
    regressor: SingleGraphTransformer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if getattr(data, "batch", None) is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    graph_alpha = global_mean_pool(alpha.to(device), data.batch)
    graph_beta  = global_mean_pool(beta.to(device),  data.batch)


    pred = regressor(data)

    labels = data.y.to(device).float().view(-1, 1)
    ce = safe_conditional_MI(graph_alpha, labels, graph_beta)

    return ce, pred

def main():  # noqa: D401
    json_path = Path("decoupled_data2.json")
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    dataset = GraphDecoupledDataset(json_path)
    print(f"Loaded {len(dataset)} samples from {json_path}")

    n_train = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=1)

    dist_s_ex, dist_d_ex, *_ = dataset[0]
    F_s, F_d = dist_s_ex.size(1), dist_d_ex.size(1)

    hidden = 64
    latent_dim = hidden // 2  # 32
    λ_reg, λ_mi, λ_ce = 0.1, 0.1, 0.005
    epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s_trans = SingleGraphTransformer(F_s, hidden).to(device)
    d_trans = SingleGraphTransformer(F_d, hidden).to(device)
    decoder = GraphDecoder(latent_dim, hidden).to(device)

    opt = optim.Adam(list(s_trans.parameters()) + list(d_trans.parameters()) + list(decoder.parameters()), lr=1e-4)
    crit = nn.MSELoss()

    for ep in range(1, epochs + 1):
        s_trans.train(); d_trans.train(); decoder.train()
        tot_loss = tot_reg = tot_mi = 0.

        for dist_s, dist_d, e_s, e_d, age in train_loader:
            dist_s, dist_d = dist_s.squeeze(0).to(device), dist_d.squeeze(0).to(device)
            e_s, e_d = e_s.squeeze(0).to(device), e_d.squeeze(0).to(device)
            age = age.to(device).float()

            opt.zero_grad()

            s_embed = s_trans.encoder(dist_s)  # (N_s,64)
            d_embed = d_trans.encoder(dist_d)
            split = latent_dim
            s_a, s_b = s_embed[:, :split], s_embed[:, split:]
            d_a, d_b = d_embed[:, :split], d_embed[:, split:]
            s_mi = calculate_MI(s_a, s_b)
            d_mi = calculate_MI(d_a, d_b)
            x_mi = calculate_MI(s_a, d_a)
            mi_loss = λ_mi * (s_mi + d_mi + x_mi)
            s_graph = Data(x=dist_s, edge_index=e_s, y=age)
            d_graph = Data(x=dist_d, edge_index=e_d, y=age)
            ce_s, pred_s = joint_uncond(s_a, s_b, s_graph, decoder, s_trans, device)
            ce_d, pred_d = joint_uncond(d_a, d_b, d_graph, decoder, d_trans, device)
            ce_loss = λ_ce * (ce_s + ce_d)
            y_true = age.unsqueeze(0).unsqueeze(1)
            reg_loss = λ_reg * (crit(pred_s, y_true) + crit(pred_d, y_true))
            loss = reg_loss + mi_loss - ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(s_trans.parameters()) + list(d_trans.parameters()) + list(decoder.parameters()), 5.)
            opt.step()

            tot_loss += loss.item(); tot_reg += reg_loss.item(); tot_mi += mi_loss.item()

        print(f"[Train] Epoch {ep:3d}/{epochs} | Loss {tot_loss/len(train_loader):.4f} | Reg {tot_reg/len(train_loader):.4f} | MI {tot_mi/len(train_loader):.4f}")
    s_trans.eval(); d_trans.eval(); decoder.eval()
    preds, gts = [], []
    with torch.no_grad():
        for dist_s, dist_d, e_s, e_d, age in test_loader:
            dist_s, dist_d = dist_s.squeeze(0).to(device), dist_d.squeeze(0).to(device)
            e_s, e_d = e_s.squeeze(0).to(device), e_d.squeeze(0).to(device)
            age = age.to(device).float()

            s_embed = s_trans.encoder(dist_s)
            d_embed = d_trans.encoder(dist_d)
            split = latent_dim
            s_a, s_b = s_embed[:, :split], s_embed[:, split:]
            d_a, d_b = d_embed[:, :split], d_embed[:, split:]

            s_graph = Data(x=dist_s, edge_index=e_s, y=age)
            d_graph = Data(x=dist_d, edge_index=e_d, y=age)
            _, p_s = joint_uncond(s_a, s_b, s_graph, decoder, s_trans, device)
            _, p_d = joint_uncond(d_a, d_b, d_graph, decoder, d_trans, device)
            preds.append(0.5 * (p_s.item() + p_d.item()))
            gts.append(age.item())

    preds, gts = np.array(preds), np.array(gts)
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))
    r2 = 1 - np.sum((preds - gts) ** 2) / np.sum((gts - gts.mean()) ** 2)
    pcc = np.corrcoef(preds, gts)[0, 1]

    print("\n[TEST] MAE {:.3f} | RMSE {:.3f} | R² {:.3f} | PCC {:.3f}".format(mae, rmse, r2, pcc))


if __name__ == "__main__":
    main()
