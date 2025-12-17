import torch
from torch_geometric.loader import DataLoader
from models.gin_sage import CryptoGNN
from loaders.graph_loader import CryptoGraphDataset


def train():

    # Load dataset
    dataset = CryptoGraphDataset(root="data/")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Labels check
    labels = [d.y.item() for d in dataset]
    print("Unique labels:", set(labels))
    print("Label counts:", {i: labels.count(i) for i in set(labels)})

    # Model: 3 classes
    model = CryptoGNN(num_features=8, num_classes=3)

    opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.8)

    for epoch in range(100):
        total, correct = 0, 0
        total_loss = 0

        for batch in loader:
            opt.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)

            loss = loss_fn(out, batch.y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        scheduler.step()

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1:03d} | Loss={total_loss:.3f} | Acc={acc:.3f}")

    torch.save(model.state_dict(), "crypto_advanced.pt")
    print("Model saved successfully!")


if __name__ == "__main__":
    train()
