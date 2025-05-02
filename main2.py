import os
import torch
import numpy as np
import torch.nn.functional as F  # 用于计算均方误差
from utils import calculate_conditional_MI, calculate_MI
from loaddata import (
    GraphJSONDataset,
    reset_model_parameters
)
from decoupling import (
    CrossGraphTranslator,
    dense_to_edge_index
)
from transformer import (
    JointGraphTransformer,
    SingleGraphTransformer
)

from causaleffect import joint_uncond, get_readout_layers
from GraphVAE import GraphEncoder, GraphDecoder, VAE_LL_loss
from torch_geometric.data import Data

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def get_adjacency_matrix(edge_index, num_nodes, device):
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def main():

    data_folder = "vector"
    demographic_csv = "HCP_demographic.csv"
    full_dataset = GraphJSONDataset(data_folder, demographic_csv)
    print(f"Loaded {len(full_dataset)} samples.")
    if len(full_dataset) == 0:
        return


    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    sample_example = full_dataset[0]
    cortex_features, fiber_features, adjacency_matrix, age = sample_example
    sMRI_input_dim = cortex_features.shape[1]
    dMRI_input_dim = fiber_features.shape[1]


    device = torch.device("cpu")
    hidden_channels = 16  


    translator = CrossGraphTranslator(
        sMRI_in_channels=sMRI_input_dim,
        dMRI_in_channels=dMRI_input_dim,
        hidden_channels=hidden_channels,
        device=device
    ).to(device)


    sMRI_transformer = SingleGraphTransformer(128, 64)
    dMRI_transformer = SingleGraphTransformer(128, 64)
    reset_model_parameters(sMRI_transformer)
    reset_model_parameters(dMRI_transformer)

    criterion = nn.MSELoss()


    sample_pretrain = train_dataset[0]
    cortex_features_pt, fiber_features_pt, adjacency_matrix_pt, _ = sample_pretrain
    num_cortex_nodes = cortex_features_pt.size(0)
    num_fiber_nodes = fiber_features_pt.size(0)
    sMRI_edge_index_pt = dense_to_edge_index(adjacency_matrix_pt[:num_cortex_nodes, :num_cortex_nodes])
    dMRI_edge_index_pt = dense_to_edge_index(adjacency_matrix_pt[num_cortex_nodes:, num_cortex_nodes:])
    sMRI_data_pt = Data(x=cortex_features_pt, edge_index=sMRI_edge_index_pt)
    dMRI_data_pt = Data(x=fiber_features_pt, edge_index=dMRI_edge_index_pt)
    adj_sMRI_pt = get_adjacency_matrix(sMRI_edge_index_pt, num_cortex_nodes, device)
    adj_dMRI_pt = get_adjacency_matrix(dMRI_edge_index_pt, num_fiber_nodes, device)
    pretrain_epochs = 500
    optimizer = torch.optim.Adam(translator.parameters(), lr=1e-3)
    print("预训练解耦模块...")
    _ = discouping_train_model(
        model=translator,  # 内部参数名仍为 model，但外部变量名为 translator
        sMRI_data=sMRI_data_pt,
        dMRI_data=dMRI_data_pt,
        adj_sMRI=adj_sMRI_pt,
        adj_dMRI=adj_dMRI_pt,
        optimizer=optimizer,
        device=device,
        epochs=pretrain_epochs,
        log_interval=100,
        namta1=0.4,
        namta2=0.1,
        namta3=0.5
    )

    translator.eval()
    for param in translator.parameters():
        param.requires_grad = False

    in_channels = sMRI_input_dim  
    latent_dim = hidden_channels  
    lambda_auto = 0.1  

    # 在联合训练前（例如在 4.3 预训练之后），你可以实例化 GraphEncoder/Decoder：
    #vae_encoder = GraphEncoder(in_channels, latent_dim, device).to(device)
    #vae_decoder = GraphDecoder(latent_dim, in_channels).to(device)
    casual_decoder = GraphDecoder(64, 64).to(device)

    joint_optimizer = optim.Adam(
        list(sMRI_transformer.parameters()) +
        list(dMRI_transformer.parameters()) +
        list(casual_decoder.parameters()),
        lr=0.0001
    )
    joint_training_epochs = 20

    for epoch in range(pretrain_epochs + 1, pretrain_epochs + joint_training_epochs + 1):
        sMRI_transformer.train()
        dMRI_transformer.train()
        #vae_encoder.train()
        #vae_decoder.train()
        casual_decoder.train()
        epoch_loss = 0.0

        for batch in train_loader:
            cortex_features_b, fiber_features_b, adjacency_matrix_b, age_b = batch
            cortex_features_b = cortex_features_b.to(device)
            fiber_features_b = fiber_features_b.to(device)
            adjacency_matrix_b = adjacency_matrix_b.to(device)
            age_b = age_b.to(device).float()  
            joint_optimizer.zero_grad()

            num_cortex = cortex_features_b.squeeze(0).size(0)
            combined_adj = adjacency_matrix_b.squeeze(0)
            cortex_adj = combined_adj[:num_cortex, :num_cortex]
            fiber_adj = combined_adj[num_cortex:, num_cortex:]

            sMRI_graph = Data(
                x=cortex_features_b.squeeze(0),
                edge_index=dense_to_edge_index(cortex_adj)
            )
            dMRI_graph = Data(
                x=fiber_features_b.squeeze(0),
                edge_index=dense_to_edge_index(fiber_adj)
            )

            with torch.no_grad():
                decoupled_reps = translator(sMRI_graph, dMRI_graph)
            sMRI_decoupled = decoupled_reps['dist_sMRI']
            dMRI_decoupled = decoupled_reps['dist_dMRI']
            sMRI_graph = Data(
                x=sMRI_decoupled,
                edge_index=dense_to_edge_index(cortex_adj),
                batch=torch.zeros(sMRI_decoupled.size(0), dtype=torch.long),
                y=age_b.clone().detach().float()  # 添加标签
            )

            dMRI_graph = Data(
                x=dMRI_decoupled,
                edge_index=dense_to_edge_index(fiber_adj),
                batch=torch.zeros(dMRI_decoupled.size(0), dtype=torch.long),
                y=age_b.clone().detach().float()  # 添加标签
            )


            split_index = sMRI_decoupled.size(1) // 2
            sMRI_alpha = sMRI_decoupled[:, :split_index]
            sMRI_beta = sMRI_decoupled[:, split_index:]
            sMRI_mi = calculate_MI(sMRI_alpha, sMRI_beta)

            causal_effect_sMRI, sMRI_pred = joint_uncond(sMRI_alpha, sMRI_beta, sMRI_graph,
                                                         casual_decoder, sMRI_transformer, device, epoch)
            loss_sMRI = criterion(sMRI_pred, age_b.unsqueeze(1))

            split_index_d = dMRI_decoupled.size(1) // 2
            dMRI_alpha = dMRI_decoupled[:, :split_index_d]
            dMRI_beta = dMRI_decoupled[:, split_index_d:]
            dMRI_mi = calculate_MI(dMRI_alpha, dMRI_beta)
            cross_mi = calculate_MI(dMRI_alpha, sMRI_alpha)
            causal_effect_dMRI,dMRI_pred = joint_uncond(dMRI_alpha, dMRI_beta, dMRI_graph,
                                                         casual_decoder, dMRI_transformer, device, epoch)
            loss_dMRI = criterion(dMRI_pred, age_b.unsqueeze(1))

            #z_vae, mu, logvar = vae_encoder(batch)
            #Xhat, _ = vae_decoder(z_vae)
            #auto_loss = VAE_LL_loss(batch.x, Xhat, logvar, mu, device)

            reg_loss = 0.1 * (loss_sMRI + loss_dMRI)
            mi_loss = 0.005 * (sMRI_mi + dMRI_mi + cross_mi)
            ce_loss = 0.005 * (causal_effect_sMRI + causal_effect_dMRI)
            loss = (reg_loss + 0.5 * mi_loss - 0.5 * ce_loss)
                    #+ lambda_auto * auto_loss)
            loss.backward()
            joint_optimizer.step()
            epoch_loss += loss.item()

            print(f"Epoch {epoch} Batch Losses: "
                  f"loss_sMRI: {loss_sMRI.item():.4f}, "
                  f"loss_dMRI: {loss_dMRI.item():.4f}, "
                  f"reg_loss: {reg_loss.item():.4f}, "
                  f"mi_loss: {mi_loss.item():.4f}, "
                  f"ce_loss: {ce_loss.item():.4f}, "
                  f"total loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Joint Training] Epoch {epoch}/{pretrain_epochs + joint_training_epochs}, Loss: {avg_loss:.4f}")

    translator.eval()
    dMRI_transformer.eval()
    sMRI_transformer.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for batch in test_loader:

            cortex_features_b, fiber_features_b, adjacency_matrix_b, age_b = batch

            cortex_features_b = cortex_features_b.to(device)
            fiber_features_b = fiber_features_b.to(device)
            adjacency_matrix_b = adjacency_matrix_b.to(device)
            age_b = age_b.to(device).float()

            num_cortex = cortex_features_b.squeeze(0).size(0)
            combined_adj = adjacency_matrix_b.squeeze(0)
            cortex_adj = combined_adj[:num_cortex, :num_cortex]
            fiber_adj = combined_adj[num_cortex:, num_cortex:]

            sMRI_graph = Data(
                x=cortex_features_b.squeeze(0),
                edge_index=dense_to_edge_index(cortex_adj)
            )
            dMRI_graph = Data(
                x=fiber_features_b.squeeze(0),
                edge_index=dense_to_edge_index(fiber_adj)
            )

            decoupled_reps = translator(sMRI_graph, dMRI_graph)
            sMRI_decoupled = decoupled_reps['dist_sMRI']
            dMRI_decoupled = decoupled_reps['dist_dMRI']

            sMRI_graph = Data(
                x=sMRI_decoupled,
                edge_index=dense_to_edge_index(cortex_adj),
                batch=torch.zeros(sMRI_decoupled.size(0), dtype=torch.long),
                y=age_b.clone().detach().float()  # 保持与训练相同结构
            )
            dMRI_graph = Data(
                x=dMRI_decoupled,
                edge_index=dense_to_edge_index(fiber_adj),
                batch=torch.zeros(dMRI_decoupled.size(0), dtype=torch.long),
                y=age_b.clone().detach().float()
            )

            split_index = sMRI_decoupled.size(1) // 2
            sMRI_alpha = sMRI_decoupled[:, :split_index]
            sMRI_beta = sMRI_decoupled[:, split_index:]

            split_index_d = dMRI_decoupled.size(1) // 2
            dMRI_alpha = dMRI_decoupled[:, :split_index_d]
            dMRI_beta = dMRI_decoupled[:, split_index_d:]

            _, sMRI_pred = joint_uncond(sMRI_alpha, sMRI_beta, sMRI_graph,
                                        casual_decoder, sMRI_transformer, device, epoch)
            _, dMRI_pred = joint_uncond(dMRI_alpha, dMRI_beta, dMRI_graph,
                                        casual_decoder, dMRI_transformer, device, epoch)

            final_pred = 0.5 * (sMRI_pred + dMRI_pred)

            predictions.append(final_pred.item())
            ground_truths.append(age_b.item())

            predictions_np = np.array(predictions)
            ground_truths_np = np.array(ground_truths)

    mae = np.mean(np.abs(predictions_np - ground_truths_np))
    mse = np.mean((predictions_np - ground_truths_np) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((predictions_np - ground_truths_np) ** 2) /
              np.sum((ground_truths_np - np.mean(ground_truths_np)) ** 2))
    corr_coef = np.corrcoef(predictions_np, ground_truths_np)[0, 1]

    print(f"Test MAE: {mae:.2f}")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.2f}")
    print(f"Test PCC: {corr_coef:.2f}")

if __name__ == "__main__":
    main()
