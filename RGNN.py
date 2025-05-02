import toimport torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.nn import MessagePassing
import geoopt
from geoopt.manifolds.poincare import PoincareBall
import pandas as pd
import numpy as np

# ---------------------------
# Causal Inference Module
# ---------------------------
class CausalInference(nn.Module):
    """
    Simple causal reasoning head: uses learned adjacency masks to compute causal effects.
    """
    def __init__(self, hidden_dim, num_vars):
        super(CausalInference, self).__init__()
        self.adj_matrix = nn.Parameter(torch.randn(num_vars, num_vars))
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch, num_vars, hidden_dim]
        mask = torch.eye(self.adj_matrix.size(0), device=x.device)
        adj = self.adj_matrix * (1 - mask)
        causal_effect = torch.einsum('ij,bjd->bid', adj, x)
        out = F.relu(self.linear(causal_effect))
        return out

# ---------------------------
# Hyperbolic GNN Layer
# ---------------------------
class HyperbolicGCN(MessagePassing):
    def __init__(self, in_dim, out_dim, manifold=None, c=1.0):
        super(HyperbolicGCN, self).__init__(aggr='add')
        self.linear = geoopt.ManifoldParameter(
            torch.randn(in_dim, out_dim), manifold=manifold)
        self.manifold = manifold or PoincareBall(c=c)

    def forward(self, x, edge_index):
        h = self.manifold.expmap0(x)
        return self.propagate(edge_index, x=h)

    def message(self, x_j):
        tangent = self.manifold.logmap0(x_j)
        transformed = tangent @ self.linear
        h = self.manifold.expmap0(transformed)
        return h

    def update(self, aggr_out):
        return aggr_out

# ---------------------------
# Drug-Disease Prediction Model
# ---------------------------
class DrugDiseasePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_vars, num_classes, c=1.0):
        super(DrugDiseasePredictor, self).__init__()
        self.manifold = PoincareBall(c=c)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.hgnn1 = HyperbolicGCN(hidden_dim, hidden_dim, manifold=self.manifold, c=c)
        self.hgnn2 = HyperbolicGCN(hidden_dim, hidden_dim, manifold=self.manifold, c=c)
        self.causal = CausalInference(hidden_dim, num_vars)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        z = F.relu(self.encoder(x))
        z = self.hgnn1(z, edge_index)
        z = self.hgnn2(z, edge_index)
        z_tangent = self.manifold.logmap0(z)
        batch_size = int(batch.max().item() + 1)
        z_batch = z_tangent.view(batch_size, -1, z_tangent.size(-1))
        causal_out = self.causal(z_batch)
        causal_flat = causal_out.view(-1, causal_out.size(-1))
        return self.classifier(causal_flat)

# ---------------------------
# PrimeKG Dataset Loader
# ---------------------------
class PrimeKGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nodes.csv', 'edges.csv', 'features.npy', 'labels.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # Place raw files manually in raw_dir

    def process(self):
        node_df = pd.read_csv(self.raw_paths[0])
        edge_df = pd.read_csv(self.raw_paths[1])
        features = np.load(self.raw_paths[2])
        labels = np.load(self.raw_paths[3])
        id_map = {nid: idx for idx, nid in enumerate(node_df['node_id'])}
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor([
            [id_map[src] for src in edge_df['source']],
            [id_map[tgt] for tgt in edge_df['target']]
        ], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ---------------------------
# Training and Inference with Metrics
# ---------------------------
if __name__ == '__main__':
    dataset = PrimeKGDataset(root='data/primekg')
    data = dataset[0]
    loader = DataLoader([data], batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrugDiseasePredictor(
        in_dim=data.num_features,
        hidden_dim=64,
        num_vars=data.num_nodes,
        num_classes=2,
        c=1.0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(1, 201):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')
    print("Training complete.")

    # Inference and Metrics
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)[:, 1]
            labels = batch.y.view(-1).cpu()
            all_scores.append(probs.cpu())
            all_labels.append(labels)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Basic Accuracy
    preds = (all_scores >= 0.5).int()
    accuracy = (preds == all_labels.int()).sum().item() / all_labels.size(0)

    # Global FP and FN Rates
    TP = ((preds == 1) & (all_labels == 1)).sum().item()
    FP = ((preds == 1) & (all_labels == 0)).sum().item()
    TN = ((preds == 0) & (all_labels == 0)).sum().item()
    FN = ((preds == 0) & (all_labels == 1)).sum().item()
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # Recall@20
    k = min(20, all_scores.size(0))
    _, idx = torch.topk(all_scores, k)
    labels_topk = all_labels[idx]
    total_pos = all_labels.sum().item()
    recall20 = labels_topk.sum().item() / total_pos if total_pos > 0 else 0.0

    # Print Metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall@20: {recall20:.4f}')
    print(f'False Positive Rate: {fpr:.4f}')
    print(f'False Negative Rate: {fnr:.4f}')


    # ---------------------------
    # Visualization of Causal Graph
    # ---------------------------
    def visualize_causal_graph(adj_param, threshold=0.0):
        adj = adj_param.detach().cpu().numpy()
        num_vars = adj.shape[0]
        G = nx.DiGraph()
        for i in range(num_vars):
            G.add_node(i)
        for i in range(num_vars):
            for j in range(num_vars):
                w = adj[i, j]
                if abs(w) > threshold:
                    G.add_edge(i, j, weight=w)
        pos = nx.circular_layout(G)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_size=500, arrowsize=20)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.coolwarm, width=2)
        plt.title('Learned Causal Adjacency')
        plt.show()


    # ---------------------------
    # SHAP Single Sample Visualization
    # ---------------------------
    def shap_single_sample(model, data, device, sample_idx=0, background_size=100):
        # Prepare data
        node_feats = data.x.cpu().numpy()
        bs = min(background_size, node_feats.shape[0])
        background_idx = np.random.choice(node_feats.shape[0], bs, replace=False)
        background = node_feats[background_idx]
        sample = node_feats[sample_idx].reshape(1, -1)

        # Prediction function for SHAP
        def predict_fn(x_matrix):
            x_tensor = torch.tensor(x_matrix, dtype=torch.float).to(device)
            preds = model(x_tensor, data.edge_index.to(device), data.batch.to(device))
            probs = F.softmax(preds, dim=1).cpu().detach().numpy()
            return probs

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(sample)

        # Plot SHAP force plot for positive class
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1],
                        feature_names=[f'feat_{i}' for i in range(sample.shape[1])])

