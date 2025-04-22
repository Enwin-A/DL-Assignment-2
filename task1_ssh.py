import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# utility: set random seeds
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# data loading & preprocessing
def load_scene(base_path, scene_id):
    nodes_path = os.path.join(base_path, f"{scene_id}.nodes")
    edges_path = os.path.join(base_path, f"{scene_id}.edges")

    nodes = pd.read_csv(
        nodes_path, header=None,
        names=[
            'original_id', 'current_x', 'current_y',
            'previous_x', 'previous_y', 'future_x', 'future_y'
        ]
    ).reset_index(drop=True)

    num_cols = ['current_x', 'current_y', 'previous_x', 'previous_y', 'future_x', 'future_y']
    nodes[num_cols] = nodes[num_cols].apply(pd.to_numeric, errors='coerce')
    valid_mask = ~nodes[['future_x', 'future_y']].isnull().any(axis=1).values

    edges = pd.read_csv(edges_path, header=None, names=['target','source'])
    edges = edges.astype(int) - 1
    edges_rev = edges.rename(columns={'target':'source','source':'target'})
    edges = pd.concat([edges, edges_rev], ignore_index=True)

    return nodes, edges, valid_mask

class SceneDataset(Dataset):
    def __init__(self, scene_ids, base_path,
                 feat_mean=None, feat_std=None,
                 targ_mean=None, targ_std=None,
                 mode='train'):
        self.scenes = []
        for sid in scene_ids:
            nodes, edges, mask = load_scene(base_path, sid)
            feats = np.nan_to_num(nodes[['current_x','current_y','previous_x','previous_y']].values)
            targs = np.nan_to_num(nodes[['future_x','future_y']].values)
            self.scenes.append({
                'feats': feats,
                'targs': targs,
                'mask': mask,
                'edges': edges
            })

        if mode == 'train':
            all_feats = np.vstack([s['feats'] for s in self.scenes])
            all_targs = np.vstack([s['targs'][s['mask']] for s in self.scenes])
            if len(all_targs) == 0:
                raise ValueError("No valid targets in training data")
            self.feat_mean = all_feats.mean(axis=0)
            self.feat_std = all_feats.std(axis=0) + 1e-6
            self.targ_mean = all_targs.mean(axis=0)
            self.targ_std = all_targs.std(axis=0) + 1e-6
        else:
            self.feat_mean = feat_mean
            self.feat_std = feat_std
            self.targ_mean = targ_mean
            self.targ_std = targ_std

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s in self.scenes:
            # normalizing features
            f = (s['feats'] - self.feat_mean) / self.feat_std
            s['features'] = torch.tensor(f, dtype=torch.float32, device=device)
            # the original targets
            s['targets_original'] = torch.tensor(s['targs'], dtype=torch.float32, device=device)
            # normalizing the targets
            t = (s['targs'] - self.targ_mean) / self.targ_std
            s['targets'] = torch.tensor(t, dtype=torch.float32, device=device)
            # mask
            s['mask'] = torch.tensor(s['mask'], dtype=torch.bool, device=device)
            # adjacency matrix
            src = s['edges']['source'].values.astype(np.int64)
            tgt = s['edges']['target'].values.astype(np.int64)
            idx = torch.tensor(np.vstack([src, tgt]), dtype=torch.long, device=device)
            vals = torch.ones(idx.shape[1], device=device)
            N = s['features'].size(0)
            s['adj'] = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

# collate
def collate_fn(batch):
    return batch[0]

# graph attention layer
class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(0)
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)
        mask = adj.to_dense() > 0
        e = torch.where(mask, e, torch.full_like(e, -9e15))
        alpha = F.softmax(e, dim=1)
        return torch.matmul(alpha, Wh)

# gat regression model
class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.gat1 = GraphAttention(in_dim, hidden_dim)
        self.dropout = nn.Dropout(0.6)
        self.gat2 = GraphAttention(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = F.relu(self.gat1(x, adj))
        x = self.dropout(x)
        return self.gat2(x, adj)

# training & evaluation
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        pred = model(data['features'], data['adj'])
        loss = criterion(pred[data['mask']], data['targets'][data['mask']])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, targ_mean, targ_std):
    model.eval()
    all_errors = []
    tm = torch.tensor(targ_mean, device=device)
    ts = torch.tensor(targ_std, device=device)
    with torch.no_grad():
        for data in loader:
            pred = model(data['features'], data['adj'])
            # denormalize
            pred_denorm = pred * ts + tm
            # mask
            mask = data['mask']
            pred_masked = pred_denorm[mask]
            targ_masked = data['targets_original'][mask]
            dists = torch.norm(pred_masked - targ_masked, dim=1)
            all_errors.append(dists.cpu().numpy())
    errs = np.concatenate(all_errors)
    mae = errs.mean()
    rmse = np.sqrt((errs**2).mean())
    pct50, pct90 = np.percentile(errs, [50, 90])
    within_1m_mm = np.mean(errs < 1000.0)   # ← 1000 mm
    return mae, rmse, pct50, pct90, within_1m_mm

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', type=str, default='gat_regressor.pth')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = [f for f in os.listdir(args.data_path) if f.endswith('.nodes')]
    scene_ids = [os.path.splitext(f)[0] for f in files]
    train_ids, test_ids = train_test_split(scene_ids, test_size=0.2, random_state=args.seed)

    train_ds = SceneDataset(train_ids, args.data_path, mode='train')
    test_ds  = SceneDataset(
        test_ids, args.data_path,
        feat_mean=train_ds.feat_mean,
        feat_std=train_ds.feat_std,
        targ_mean=train_ds.targ_mean,
        targ_std=train_ds.targ_std,
        mode='test'
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds , batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = GATRegressor(in_dim=4, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_mae = float('inf')
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}")
        mae, rmse, p50, p90, w1 = evaluate(model, test_loader, train_ds.targ_mean, train_ds.targ_std)
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), args.save_model)

    mae, rmse, p50, p90, w1 = evaluate(model, test_loader, train_ds.targ_mean, train_ds.targ_std)
    print(f"Test MAE: {mae:.4f} mm,  RMSE: {rmse:.4f} mm")
    print(f"Median / 90th pct: {p50:.4f}, {p90:.4f} mm")
    print(f"Fraction within 1 m: {w1:.2%}")
    print(f"Best model saved to {args.save_model}")
