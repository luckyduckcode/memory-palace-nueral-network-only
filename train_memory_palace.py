import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Import our neural network models
from mnemonic_model import (
    DIM_Net, GenerativeMnemonicModel, TRHD_MnemonicMapper,
    geometric_loss, compression_vividness_loss, truth_consistency_loss,
    ChessCubeLattice
)

# --- 1. CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Restored to 50
SAVE_INTERVAL = 10

# Model dimensions
SEMANTIC_DIM = 768
MNEMONIC_DIM = 256
OUTPUT_DIM = 3
TOTAL_INPUT_DIM = 71  # TRHD input dimension
NUM_FACTS_PER_CLUSTER = 10

# --- 2. DATA LOADING ---

class FactClusterDataset(Dataset):
    """Dataset for training with fact clusters"""

    def __init__(self, clusters_df):
        self.clusters_df = clusters_df
        # Group by base cluster (remove _f suffix)
        clusters_df['base_cluster'] = clusters_df['cluster_id'].str.replace(r'_f\d+$', '', regex=True)
        self.cluster_groups = clusters_df.groupby('base_cluster')

        # Create cluster data
        self.cluster_data = []
        for base_cluster, group in self.cluster_groups:
            facts = group['math_concept'].tolist()
            if len(facts) == NUM_FACTS_PER_CLUSTER:
                cluster_info = {
                    'cluster_id': base_cluster,
                    'facts': facts,
                    'target_coords': torch.tensor([
                        group['x_coord'].iloc[0],
                        group['y_coord'].iloc[0],
                        group['z_coord'].iloc[0]
                    ], dtype=torch.float32),
                    'color_parity': group['color_parity'].iloc[0]
                }
                self.cluster_data.append(cluster_info)

    def __len__(self):
        return len(self.cluster_data)

    def __getitem__(self, idx):
        cluster = self.cluster_data[idx]

        # Create input features for TRHD_MnemonicMapper
        # This is a simplified version - in practice you'd use embeddings
        input_features = []
        for fact in cluster['facts']:
            # Simple text length and position features (placeholder for real embeddings)
            fact_vector = torch.randn(TOTAL_INPUT_DIM - 1)  # Random for now
            fact_vector[0] = len(fact) / 100.0  # Normalized length
            truth_score = torch.rand(1) * 0.5 + 0.5  # Simulated truth score
            fact_vector = torch.cat([fact_vector, truth_score])
            input_features.append(fact_vector)

        return {
            'cluster_input': torch.stack(input_features),  # (10, 71)
            'target_coords': cluster['target_coords'],     # (3,)
            'color_parity': cluster['color_parity'],
            'cluster_id': cluster['cluster_id']
        }

# --- 3. TRAINING FUNCTIONS ---

def train_dim_net(model, dataloader, optimizer, criterion, epoch):
    """Train DIM-Net on individual facts (simplified)"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        cluster_input = batch['cluster_input'].to(DEVICE)  # (batch, 10, 71)
        target_coords = batch['target_coords'].to(DEVICE)  # (batch, 3)

        # For DIM-Net training, we'll use simplified inputs
        # In practice, you'd have individual fact embeddings
        batch_size = cluster_input.size(0)
        dummy_semantic = torch.randn(batch_size, SEMANTIC_DIM).to(DEVICE)

        optimizer.zero_grad()
        predicted_coords = model(dummy_semantic)
        loss = criterion(predicted_coords, target_coords)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_trhd_mapper(model, dataloader, optimizer, epoch):
    """Train TRHD_MnemonicMapper on fact clusters"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        cluster_input = batch['cluster_input'].to(DEVICE)  # (batch, 10, 71)
        target_coords = batch['target_coords'].to(DEVICE)  # (batch, 3)
        color_parity = batch['color_parity'].to(DEVICE)

        optimizer.zero_grad()
        predicted_coords = model(cluster_input)

        # Combined loss: geometric + truth consistency
        geo_loss = geometric_loss(predicted_coords, target_coords)
        truth_loss = truth_consistency_loss(predicted_coords, color_parity.unsqueeze(-1),
                                          torch.rand(predicted_coords.size(0), 1).to(DEVICE))
        loss = geo_loss + 0.1 * truth_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_models(dim_net, trhd_mapper, dataloader):
    """Validate both models"""
    dim_net.eval()
    trhd_mapper.eval()

    dim_losses = []
    trhd_losses = []

    with torch.no_grad():
        for batch in dataloader:
            cluster_input = batch['cluster_input'].to(DEVICE)
            target_coords = batch['target_coords'].to(DEVICE)
            color_parity = batch['color_parity'].to(DEVICE)

            # DIM-Net validation
            batch_size = cluster_input.size(0)
            dummy_semantic = torch.randn(batch_size, SEMANTIC_DIM).to(DEVICE)
            predicted_coords_dim = dim_net(dummy_semantic)
            dim_loss = geometric_loss(predicted_coords_dim, target_coords)
            dim_losses.append(dim_loss.item())

            # TRHD validation
            predicted_coords_trhd = trhd_mapper(cluster_input)
            geo_loss = geometric_loss(predicted_coords_trhd, target_coords)
            truth_loss = truth_consistency_loss(predicted_coords_trhd,
                                              color_parity.unsqueeze(-1),
                                              torch.rand(predicted_coords_trhd.size(0), 1).to(DEVICE))
            trhd_loss = geo_loss + 0.1 * truth_loss
            trhd_losses.append(trhd_loss.item())

    return np.mean(dim_losses), np.mean(trhd_losses)

# --- 4. MAIN TRAINING LOOP ---

def main():
    print("=== AI-POWERED MEMORY PALACE TRAINING ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Training on {len(pd.read_csv('math_training_data.csv'))} mathematical concepts in clusters", flush=True)

    # Load cluster data
    clusters_df = pd.read_csv('math_training_data.csv')
    dataset = FactClusterDataset(clusters_df)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset: {len(dataset)} clusters, {len(dataloader)} batches", flush=True)

    # Initialize models
    dim_net = DIM_Net().to(DEVICE)
    trhd_mapper = TRHD_MnemonicMapper().to(DEVICE)

    # Optimizers
    dim_optimizer = optim.Adam(dim_net.parameters(), lr=LEARNING_RATE)
    trhd_optimizer = optim.Adam(trhd_mapper.parameters(), lr=LEARNING_RATE)

    # Loss functions
    dim_criterion = geometric_loss

    # Training history
    history = {
        'epoch': [],
        'dim_train_loss': [],
        'trhd_train_loss': [],
        'dim_val_loss': [],
        'trhd_val_loss': []
    }

    print("\n=== STARTING TRAINING ===", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}", flush=True)

        # Train models
        dim_train_loss = train_dim_net(dim_net, dataloader, dim_optimizer, dim_criterion, epoch)
        trhd_train_loss = train_trhd_mapper(trhd_mapper, dataloader, trhd_optimizer, epoch)

        # Validate
        dim_val_loss, trhd_val_loss = validate_models(dim_net, trhd_mapper, dataloader)

        # Record history
        history['epoch'].append(epoch)
        history['dim_train_loss'].append(dim_train_loss)
        history['trhd_train_loss'].append(trhd_train_loss)
        history['dim_val_loss'].append(dim_val_loss)
        history['trhd_val_loss'].append(trhd_val_loss)

        print(".4f", flush=True)
        print(".4f", flush=True)

        # Save checkpoints
        if epoch % SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'dim_net_state_dict': dim_net.state_dict(),
                'trhd_mapper_state_dict': trhd_mapper.state_dict(),
                'dim_optimizer_state_dict': dim_optimizer.state_dict(),
                'trhd_optimizer_state_dict': trhd_optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pth", flush=True)

    # Save final models
    torch.save(dim_net.state_dict(), 'dim_net_final.pth')
    torch.save(trhd_mapper.state_dict(), 'trhd_mapper_final.pth')

    # Save training history
    pd.DataFrame(history).to_csv('training_history.csv', index=False)

    print("\n=== TRAINING COMPLETE ===")
    print("Models saved: dim_net_final.pth, trhd_mapper_final.pth")
    print("Training history: training_history.csv")

    # Final evaluation
    final_dim_loss, final_trhd_loss = validate_models(dim_net, trhd_mapper, dataloader)
    print(".4f")
    print(".4f")

if __name__ == "__main__":
    main()