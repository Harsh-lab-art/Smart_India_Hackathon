import torch
from torch_geometric.data import Data
from models.gin_sage import CryptoGNN

# 1. Load Model
model = CryptoGNN()
model.load_state_dict(torch.load("crypto_advanced.pt", map_location="cpu"))
model.eval()

# 2. Example Graph Input (put any dummy graph for testing)
# 10 nodes → each node has 8 features
x = torch.rand(10, 8)

# edges (random example) — must be [2, num_edges] shape
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6]
], dtype=torch.long)

# 3. Wrap in PyG Data format
data = Data(x=x, edge_index=edge_index)

# 4. Do prediction
with torch.no_grad():
    logits = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long))
    pred = logits.argmax(dim=1).item()

print("Model raw output:", logits)
print("Predicted class:", pred)
