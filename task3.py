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

# ------------------------
# Utility: set random seeds
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------
# Data loading & preprocessing (with self-loops)
# ------------------------
def load_scene(base_path, scene_id):
    nodes = pd.read_csv(
        os.path.join(base_path, f"{scene_id}.nodes"), header=None,
        names=[
            'original_id', 'current_x', 'current_y',
            'previous_x', 'previous_y', 'future_x', 'future_y'
        ]
    ).reset_index(drop=True)
    num_cols = ['current_x','current_y','previous_x','previous_y','future_x','future_y']
    nodes[num_cols] = nodes[num_cols].apply(pd.to_numeric, errors='coerce')
    valid = ~nodes[['future_x','future_y']].isnull().any(axis=1).values
    edges = pd.read_csv(
        os.path.join(base_path, f"{scene_id}.edges"), header=None,
        names=['target','source']
    ).astype(int) - 1
    # undirected
    edges_rev = edges.rename(columns={'target':'source','source':'target'})
    edges = pd.concat([edges, edges_rev], ignore_index=True)
    return nodes, edges, valid

class SceneDataset(Dataset):
    def __init__(self, scene_ids, base_path, feat_mean=None, feat_std=None,
                 disp_mean=None, disp_std=None, mode='train'):
        self.scenes = []
        for sid in scene_ids:
            nodes, edges, mask = load_scene(base_path, sid)
            feats = np.nan_to_num(nodes[['current_x','current_y','previous_x','previous_y']].values)
            curr = nodes[['current_x','current_y']].values
            fut  = nodes[['future_x','future_y']].values
            disp = np.nan_to_num(fut - curr)
            self.scenes.append({'feats':feats,'disp':disp,'mask':mask,
                                 'edges':edges,'curr':curr,'fut':fut})
        # normalization
        if mode=='train':
            all_feats = np.vstack([s['feats'] for s in self.scenes])
            all_disp  = np.vstack([s['disp'][s['mask']] for s in self.scenes])
            self.feat_mean = all_feats.mean(axis=0)
            self.feat_std  = all_feats.std(axis=0) + 1e-6
            self.disp_mean = all_disp.mean(axis=0)
            self.disp_std  = all_disp.std(axis=0) + 1e-6
        else:
            self.feat_mean = feat_mean
            self.feat_std  = feat_std
            self.disp_mean = disp_mean
            self.disp_std  = disp_std
        # to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s in self.scenes:
            # normalize
            s['features'] = torch.tensor((s['feats'] - self.feat_mean)/self.feat_std,
                                         dtype=torch.float32, device=device)
            s['disp_norm'] = torch.tensor((s['disp'] - self.disp_mean)/self.disp_std,
                                          dtype=torch.float32, device=device)
            s['mask'] = torch.tensor(s['mask'], dtype=torch.bool, device=device)
            s['curr_xy'] = torch.tensor(s['curr'], dtype=torch.float32, device=device)
            s['fut_xy']  = torch.tensor(s['fut'], dtype=torch.float32, device=device)
            # build sparse adjacency with self-loops
            src = s['edges']['source'].values.astype(np.int64)
            tgt = s['edges']['target'].values.astype(np.int64)
            N = s['features'].size(0)
            # include self-loops
            self_loop = np.arange(N, dtype=np.int64)
            src = np.concatenate([src, self_loop])
            tgt = np.concatenate([tgt, self_loop])
            idx = torch.tensor(np.vstack([src,tgt]), dtype=torch.long, device=device)
            vals = torch.ones(idx.shape[1], device=device)
            s['adj'] = torch.sparse_coo_tensor(idx, vals, (N,N)).coalesce()
    def __len__(self): return len(self.scenes)
    def __getitem__(self, idx): return self.scenes[idx]

def collate_fn(batch): return batch[0]

# ------------------------
# Cosine-Based Attention
# ------------------------
class CosineAttention(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-8):
        super().__init__()
        # optional linear projection
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.eps = eps

    def forward(self, h, adj):
        Wh = self.W(h)                # (N, out_dim)
        N = Wh.size(0)
        norms = Wh.norm(p=2, dim=1, keepdim=True)            # (N,1)
        sim = (Wh @ Wh.t()) / (norms * norms.t() + self.eps)  # (N,N)
        # build mask, include self-loops to avoid isolated rows
        dense_adj = adj.to_dense()
        mask = dense_adj > 0
        mask = mask | torch.eye(N, device=mask.device, dtype=torch.bool)
        # mask similarity
        sim_masked = torch.where(mask, sim, torch.full_like(sim, float('-inf')))
        alpha = F.softmax(sim_masked, dim=1)
        return alpha @ Wh

class MultiHeadCosine(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            CosineAttention(in_dim, out_dim) for _ in range(num_heads)
        ])
    def forward(self, h, adj):
        out = [head(h, adj) for head in self.heads]
        return torch.stack(out, dim=0).mean(dim=0)

# ------------------------
# GAT Regressor with Cosine Attention
# ------------------------
class GATCosineRegressor(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads, deeper_embed=False):
        super().__init__()
        if deeper_embed:
            self.embed = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            in_dim = hidden_dim
        else:
            self.embed = nn.Identity()
            in_dim = feature_dim
        self.attn1 = MultiHeadCosine(in_dim, hidden_dim, num_heads)
        self.dropout = nn.Dropout(0.6)
        self.attn2 = MultiHeadCosine(hidden_dim, 2, num_heads)
    def forward(self, x, adj):
        x = self.embed(x)
        x = F.relu(self.attn1(x, adj))
        x = self.dropout(x)
        return self.attn2(x, adj)

# ------------------------
# Training & Evaluation
# ------------------------
def train_epoch(model, loader, opt, crit):
    model.train()
    total = 0.0
    for data in loader:
        opt.zero_grad()
        pred = model(data['features'], data['adj'])
        loss = crit(pred[data['mask']], data['disp_norm'][data['mask']])
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, disp_mean, disp_std):
    model.eval()
    all_errs = []
    dm = torch.tensor(disp_mean, device=device)
    ds = torch.tensor(disp_std, device=device)
    with torch.no_grad():
        for data in loader:
            pred = model(data['features'], data['adj'])
            disp = pred * ds + dm
            abs_pred = data['curr_xy'] + disp
            m = data['mask']
            d = torch.norm(abs_pred[m] - data['fut_xy'][m], dim=1)
            all_errs.append(d.cpu().numpy())
    errs = np.concatenate(all_errs)
    return errs.mean(), np.sqrt((errs**2).mean()), *np.percentile(errs, [50, 90])

# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    type=str,   default='dataset/dataset')
    parser.add_argument('--hidden_dim',   type=int,   default=64)
    parser.add_argument('--num_heads',    type=int,   default=2)
    parser.add_argument('--deeper_embed', action='store_true')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load scenes
    files = [f for f in os.listdir(args.data_path) if f.endswith('.nodes')]
    scene_ids = [os.path.splitext(f)[0] for f in files]
    train_ids, test_ids = train_test_split(scene_ids, test_size=0.2, random_state=args.seed)

    train_ds = SceneDataset(train_ids, args.data_path, mode='train')
    test_ds  = SceneDataset(test_ids, args.data_path,
                            feat_mean=train_ds.feat_mean,
                            feat_std=train_ds.feat_std,
                            disp_mean=train_ds.disp_mean,
                            disp_std=train_ds.disp_std,
                            mode='test')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn)

    # model, optimizer, loss
    model = GATCosineRegressor(4, args.hidden_dim, args.num_heads, args.deeper_embed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # train
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}")

    # eval
    mae, rmse, m50, m90 = evaluate(model, test_loader, train_ds.disp_mean, train_ds.disp_std)
    print(f"Test MAE: {mae:.4f} mm, RMSE: {rmse:.4f} mm")
    print(f"Median / 90th pct: {m50:.4f}, {m90:.4f} mm")
    # save
    torch.save(model.state_dict(), 'task3_cosine_model.pth')
    print("Model saved to task3_cosine_model.pth")
