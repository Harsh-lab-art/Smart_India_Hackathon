import torch
import numpy as np
from preprocessing.code_to_vector import CodeToVector
from models.gin_sage import CryptoGNN

def classify(code):
    extractor = CodeToVector()
    vec = extractor.get_vector(code)

    x = torch.tensor([vec[:4]], dtype=torch.float)
    edge_index = torch.tensor([[0],[0]])
    batch = torch.tensor([0])

    model = CryptoGNN(num_classes=3)
    model.load_state_dict(torch.load("crypto_advanced.pt"))
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index, batch)
        prob = torch.softmax(out, 1)[0]

    return {
        "NonCrypto": float(prob[0]),
        "AES128": float(prob[1]),
        "SHA256": float(prob[2])
    }
