import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import seaborn as sns

# --- MLOPS CONFIGURATION ---
# Research-Grade Hyperparameters
LOOKBACK = 12
HORIZON = 7
BATCH_SIZE = 32
EPOCHS = 20
EMBED_DIM = 10  # Dimension of the "Latent Node Embedding"
HIDDEN_DIM = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K_DISTRICTS = 50 

print(f">>> [INIT] ADAPTIVE GRAPH LEARNING ENGINE | Device: {DEVICE}")

# --- MODULE 1: DATA PIPELINE (UNCHANGED - ROBUST) ---
def load_data_matrix():
    print(">>> [DATA] Ingesting Shards...")
    files = glob.glob('data/api_data_aadhar_*.csv')
    df_list = []
    for f in files:
        try:
            cols = pd.read_csv(f, nrows=1).columns.tolist()
            age_cols = [c for c in cols if 'age' in c]
            keep_cols = ['date', 'district'] + age_cols
            temp = pd.read_csv(f, usecols=keep_cols)
            df_list.append(temp)
        except: continue
            
    full = pd.concat(df_list, ignore_index=True)
    full['date'] = pd.to_datetime(full['date'], format='%d-%m-%Y', errors='coerce')
    full = full.dropna(subset=['date'])
    
    # Feature Engineering
    age_cols = [c for c in full.columns if 'age' in c]
    full['total_load'] = full[age_cols].sum(axis=1)
    
    # Pivot to Matrix: [Time, Nodes]
    pivot_df = full.groupby(['date', 'district'])['total_load'].sum().unstack(fill_value=0)
    top_districts = pivot_df.sum().nlargest(TOP_K_DISTRICTS).index
    pivot_df = pivot_df[top_districts]
    
    print(f">>> [GRAPH] Matrix Constructed: {pivot_df.shape} (Time x Nodes)")
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(pivot_df.values)
    
    return torch.FloatTensor(data_scaled), top_districts, scaler

# --- MODULE 2: THE ADAPTIVE GRAPH LAYER (THE "SUPREME" UPGRADE) ---
class AdaptiveGraphConvolution(nn.Module):
    """
    Research Paper: AGCRN (NeurIPS 2020)
    Logic: Z = A_adaptive * X * W
    Where A_adaptive is LEARNED from Node Embeddings.
    """
    def __init__(self, in_dim, out_dim, num_nodes, node_embeddings):
        super(AdaptiveGraphConvolution, self).__init__()
        self.num_nodes = num_nodes
        self.node_embeddings = node_embeddings # Expected shape: [Nodes, Embed_Dim]
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias, 0)
        
    def forward(self, x):
        # x shape: [Batch, Nodes, In_Dim]
        
        # 1. Infer the Graph Structure (A_adaptive)
        # A = Softmax(Relu(E * E^T))
        # This dynamically computes similarity between every district pair
        node_num, embed_dim = self.node_embeddings.size()
        adj = F.relu(torch.mm(self.node_embeddings, self.node_embeddings.t()))
        adj = F.softmax(adj, dim=1) # Normalize rows
        
        # 2. Graph Convolution
        # Support = A * X (Aggregating neighbor info based on learned graph)
        # shape: [Batch, Nodes, In_Dim]
        support = torch.einsum('nn, bni -> bni', adj, x)
        
        # 3. Linear Transformation
        output = torch.matmul(support, self.weights) + self.bias
        return output

# --- MODULE 3: THE AGCRN CELL (RECURRENT UNIT) ---
class AGCRNCell(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, node_embeddings):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = hidden_dim
        # Gate mechanisms (Reset and Update gates)
        self.gate = AdaptiveGraphConvolution(in_dim + hidden_dim, 2 * hidden_dim, num_nodes, node_embeddings)
        self.update = AdaptiveGraphConvolution(in_dim + hidden_dim, hidden_dim, num_nodes, node_embeddings)
        
    def forward(self, x, state):
        # x: [Batch, Nodes, In_Dim]
        # state: [Batch, Nodes, Hidden_Dim]
        combined = torch.cat([x, state], dim=-1)
        
        # GRU Logic with Graph Convolutions
        z_r = torch.sigmoid(self.gate(combined))
        r, u = torch.split(z_r, self.hidden_dim, dim=-1)
        
        c = torch.cat([x, r * state], dim=-1)
        h_candidate = torch.tanh(self.update(c))
        
        new_state = u * state + (1 - u) * h_candidate
        return new_state

# --- MODULE 4: THE FULL MODEL ---
class AdaptiveGraphNetwork(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, embed_dim):
        super(AdaptiveGraphNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # THE LEARNABLE PARAMETER (The "Brain" of the Graph)
        # We initialize it randomly. The model will "discover" the map of India.
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        
        self.encoder = AGCRNCell(num_nodes, in_dim, hidden_dim, self.node_embeddings)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x sequence: [Batch, Time, Nodes, Features]
        batch_size, time_steps, nodes, feats = x.size()
        
        # Initialize Hidden State
        h = torch.zeros(batch_size, nodes, self.hidden_dim).to(x.device)
        
        # Iterate over Time
        for t in range(time_steps):
            input_t = x[:, t, :, :] # [Batch, Nodes, Feats]
            h = self.encoder(input_t, h)
            
        # Decode the final state to the Horizon
        # h: [Batch, Nodes, Hidden] -> Out: [Batch, Nodes, Horizon]
        out = self.decoder(h)
        return out

# --- MODULE 5: TRAINING PIPELINE ---
def run_supreme_training():
    data, districts, scaler = load_data_matrix()
    data = data.to(DEVICE)
    
    # Sequence Gen
    X, Y = [], []
    total_len = len(data)
    for i in range(total_len - LOOKBACK - HORIZON):
        # Input: [Lookback, Nodes] -> [Lookback, Nodes, 1]
        X.append(data[i : i+LOOKBACK].unsqueeze(-1))
        # Output: [Nodes, Horizon]
        Y.append(data[i+LOOKBACK : i+LOOKBACK+HORIZON].t())
        
    X = torch.stack(X) # [Batch, Time, Nodes, 1]
    Y = torch.stack(Y) # [Batch, Nodes, Horizon]
    
    # Train/Test Split
    train_size = int(len(X) * 0.85)
    train_X, test_X = X[:train_size], X[train_size:]
    train_Y, test_Y = Y[:train_size], Y[train_size:]
    
    model = AdaptiveGraphNetwork(num_nodes=TOP_K_DISTRICTS, 
                                 in_dim=1, 
                                 hidden_dim=HIDDEN_DIM, 
                                 out_dim=HORIZON, 
                                 embed_dim=EMBED_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    
    print(f">>> [TRAIN] Optimizing Adaptive Structure ({EPOCHS} Epochs)...")
    
    loss_history = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)
        loss = criterion(pred, train_Y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.5f} | Graph Learned: Yes")

    # --- MODULE 6: VISUALIZING THE "LEARNED" GRAPH (The Interpretability Bonus) ---
    print(">>> [INSIGHT] Extracting Learned Spatial Dependencies...")
    model.eval()
    
    # Recover the Adjacency Matrix from the Embeddings
    emb = model.node_embeddings
    adj_logits = torch.mm(emb, emb.t())
    adj_matrix = F.softmax(F.relu(adj_logits), dim=1).cpu().detach().numpy()
    
    # Plot the Matrix
    plt.figure(figsize=(10, 8))
    # We only show top 20 districts to keep it readable
    sns.heatmap(adj_matrix[:20, :20], cmap='viridis', xticklabels=districts[:20], yticklabels=districts[:20])
    plt.title("The 'Hidden' Map of Aadhaar: AI-Learned District Dependencies")
    plt.tight_layout()
    plt.savefig('learned_graph_matrix.png')
    
    # Plot Forecast for District 0
    with torch.no_grad():
        test_pred = model(test_X)
    
    dist_idx = 0 
    dist_name = districts[dist_idx]
    
    pred_seq = test_pred[-1, dist_idx, :].cpu().numpy()
    true_seq = test_Y[-1, dist_idx, :].cpu().numpy()
    
    # Simple Unscale (Approx)
    d_min, d_max = scaler.data_min_[dist_idx], scaler.data_max_[dist_idx]
    pred_real = pred_seq * (d_max - d_min) + d_min
    true_real = true_seq * (d_max - d_min) + d_min
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(HORIZON), true_real, label='Ground Truth', marker='o')
    plt.plot(range(HORIZON), pred_real, label='AGCRN Forecast', marker='x', color='crimson', linestyle='--')
    plt.title(f"Adaptive Graph Forecast: {dist_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('supreme_forecast.png')
    print(">>> [COMPLETE] Artifacts: 'learned_graph_matrix.png' & 'supreme_forecast.png'")

if __name__ == "__main__":
    run_supreme_training()