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
import matplotlib.pyplot as plt

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
# Data loading & preprocessing
# ------------------------
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
    valid = ~nodes[['future_x','future_y']].isnull().any(axis=1).values

    edges = pd.read_csv(edges_path, header=None, names=['target','source'])
    edges = edges.astype(int) - 1
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
            self.scenes.append({'feats':feats,'disp':disp,'mask':mask,'edges':edges,'curr':curr,'fut':fut})

        # compute normalization statistics
        if mode == 'train':
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s in self.scenes:
            f = (s['feats'] - self.feat_mean) / self.feat_std
            s['features'] = torch.tensor(f, dtype=torch.float32, device=device)
            s['disp_norm'] = torch.tensor((s['disp'] - self.disp_mean) / self.disp_std,
                                          dtype=torch.float32, device=device)
            s['mask'] = torch.tensor(s['mask'], dtype=torch.bool, device=device)
            s['curr_xy'] = torch.tensor(s['curr'], dtype=torch.float32, device=device)
            s['fut_xy']  = torch.tensor(s['fut'],  dtype=torch.float32, device=device)
            src = s['edges']['source'].values.astype(np.int64)
            tgt = s['edges']['target'].values.astype(np.int64)
            idx = torch.tensor(np.vstack([src,tgt]), dtype=torch.long, device=device)
            vals = torch.ones(idx.size(1), device=device)
            N = s['features'].size(0)
            s['adj'] = torch.sparse_coo_tensor(idx, vals, (N,N)).coalesce()

    def __len__(self): return len(self.scenes)
    def __getitem__(self, idx): return self.scenes[idx]

def collate_fn(batch): return batch[0]

# ------------------------
# Graph Attention & Multi-Head
# ------------------------
class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(0)
        Wh_i = Wh.unsqueeze(1).expand(N,N,-1)
        Wh_j = Wh.unsqueeze(0).expand(N,N,-1)
        e = self.leaky(self.a(torch.cat([Wh_i,Wh_j],dim=-1))).squeeze(-1)
        mask = adj.to_dense() > 0
        e = torch.where(mask, e, torch.full_like(e, -9e15))
        alpha = F.softmax(e, dim=1)
        return torch.matmul(alpha, Wh)

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttention(in_dim, out_dim)
                                     for _ in range(num_heads)])

    def forward(self, h, adj):
        out = [head(h, adj) for head in self.heads]
        return torch.stack(out, dim=0).mean(dim=0)

# ------------------------
# GAT Regressor
# ------------------------
class GATRegressor(nn.Module):
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
        self.gat1 = MultiHeadAttention(in_dim, hidden_dim, num_heads)
        self.dropout = nn.Dropout(0.6)
        self.gat2 = MultiHeadAttention(hidden_dim, 2, num_heads)

    def forward(self, x, adj):
        x = self.embed(x)
        x = F.relu(self.gat1(x, adj))
        x = self.dropout(x)
        return self.gat2(x, adj)

# ------------------------
# Train & Eval
# ------------------------
def train_epoch(model, loader, opt, crit):
    model.train()
    total = 0
    for data in loader:
        opt.zero_grad()
        pred = model(data['features'], data['adj'])
        loss = crit(pred[data['mask']], data['disp_norm'][data['mask']])
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)


def evaluate(model, loader, disp_mean, disp_std):
    model.eval()
    errs = []
    dm = torch.tensor(disp_mean, device=device)
    ds = torch.tensor(disp_std, device=device)
    with torch.no_grad():
        for data in loader:
            pred = model(data['features'], data['adj'])
            pred_disp = pred * ds + dm
            abs_pred = data['curr_xy'] + pred_disp
            m = data['mask']
            d = torch.norm(abs_pred[m] - data['fut_xy'][m], dim=1)
            errs.append(d.cpu().numpy())
    arr = np.concatenate(errs)
    return arr.mean(), np.sqrt((arr**2).mean()), *np.percentile(arr,[50,90])

# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    type=str,   default='dataset/dataset')
    parser.add_argument('--hidden_dim',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--grid_search',  action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # scenes
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

    # grid or single
    results = []
    head_options  = [1, 2, 4]
    embed_options = [False, True]

    combos = []
    if args.grid_search:
        combos = [(h, de) for h in head_options for de in embed_options]
    else:
        combos = [(1, False)]  # default single run

    for num_heads, deeper in combos:
        model = GATRegressor(4, args.hidden_dim, num_heads, deeper).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit  = nn.MSELoss()
        for epoch in range(1, args.epochs+1):
            train_epoch(model, train_loader, opt, crit)
        mae, rmse, m50, m90 = evaluate(model, test_loader,
                                       train_ds.disp_mean, train_ds.disp_std)
        results.append({
            'num_heads': num_heads,
            'deeper_embed': deeper,
            'MAE': mae,
            'RMSE': rmse,
            'Median': m50,
            '90th_pct': m90
        })
        print(f"Heads={num_heads}, embed={deeper} -> MAE={mae:.4f}, RMSE={rmse:.4f}")

    # save results CSV
    df = pd.DataFrame(results)
    csv_path = 'grid_search_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # --------------
    # Plot & save charts
    # --------------
    # Pivot and plot for MAE and RMSE
    for metric in ['MAE', 'RMSE']:
        pivot = df.pivot(index='num_heads', columns='deeper_embed', values=metric)
        ax = pivot.plot(kind='bar')
        ax.set_xlabel('Number of Attention Heads')
        ax.set_ylabel(metric)
        ax.set_title(f'Task 2: {metric} by #Heads and Deeper Embedding')
        plt.tight_layout()
        fig_path = f'{metric.lower()}_by_heads_embed.png'
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
        plt.clf()

    # Optionally, plot median and 90th percentile too
    for metric in ['Median', '90th_pct']:
        pivot = df.pivot(index='num_heads', columns='deeper_embed', values=metric)
        ax = pivot.plot(kind='bar')
        ax.set_xlabel('Number of Attention Heads')
        ax.set_ylabel(metric)
        ax.set_title(f'Task 2: {metric} by #Heads and Deeper Embedding')
        plt.tight_layout()
        fig_path = f'{metric.lower()}_by_heads_embed.png'
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
        plt.clf()
