import torch
from torch import nn
from .CLIP import CLIPExtractor 
from .GATv2 import GATv2
from graph.builder import build_graph_from_patches_coords
from torch_geometric.data import Batch

"""
    Klasa zarządzająca modelem klasyfikatora składającego się z ekstraktora cech CLIP, grafowego modelu GNN (GATv2) oraz head (głowy klasyfikatora).
    PARAMETRY:
        cfg - konfiguracja modelu, zawiera parametry ekstraktora CLIP, GNN oraz inne ustawienia
    PROCEDURY:
        forward_single() - przetwarza pojedynczy obraz podzielony na patche. Ekstrahuje cechy patchy za pomocą CLIP, buduje graf z patchy i ich koordynatów, zwraca graf
        forward() - przetwarza batch obrazów, wywołując forward_single() dla każdego z nich, który zwraca graf cech patchy, łączy wyniki w batch i przekazuje przez GNN 
        oraz head, zwraca logits (wyniki klasyfikacji)
"""
class GraphClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.clip_extr =  CLIPExtractor(cfg.model.clip_name, cfg.model.clip_pretrained)
        in_dim = self.clip_extr.out_dim + 2  # doklejamy 2 wartości dla coord's
        self.gnn = GATv2(in_dim, cfg.model.gat.hidden_dim, cfg.model.gat.out_dim, cfg.model.gat.heads1, cfg.model.gat.heads2, cfg.model.gat.dropout)
        self.cfg = cfg

    def forward_single(self, patches, coords):
        with torch.set_grad_enabled(self.cfg.model.clip_trainable):
            emb = self.clip_extr.extract_features(patches)

        data = build_graph_from_patches_coords(emb, coords, k=self.cfg.graph.knn_k)
        assert data.x.ndim == 2 and data.edge_index.ndim == 2
        assert int(data.edge_index.max()) < data.x.size(0)

        return data

    def forward(self, batch):
        outputs = []
        for sample in batch:
            patches = sample["patches"].to(self.device)
            coords = sample["coords"].to(self.device)
            coords = coords / coords.max()
            outputs.append(self.forward_single(patches,coords))
        batch = Batch.from_data_list(outputs).to(self.device)

        logits = self.gnn(batch)
        return logits
    