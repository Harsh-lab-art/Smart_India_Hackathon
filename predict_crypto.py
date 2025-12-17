import argparse
import torch
import torch.nn.functional as F
from models.gin_sage import CryptoGNN
import os

# -----------------------------
# Process file function
# -----------------------------
def process_source_code(file_path):
    """Reads the file and returns its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Dummy graph conversion
# (Replace with your real logic)
# -----------------------------
def convert_code_to_graph(code):
    """
    Placeholder: Convert source code → graph input.
    Replace with your real AST/graph logic.
    """
    import torch
    from torch_geometric.data import Data

    # Dummy node features (8 features) and simple edges
    x = torch.rand((10, 8))
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch)


# -----------------------------
# Print final result
# -----------------------------
def print_result(function_name, detected_algo, confidence):
    print("\n--- RESULT ---")
    print(f"Function: {function_name}")
    print(f"Detected: {detected_algo}")
    print(f"Confidence: {confidence:.1f}%")


# -----------------------------
# Main Prediction Pipeline
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    file_path = args.file

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Load code
    source_code = process_source_code(file_path)

    # Convert to graph
    graph = convert_code_to_graph(source_code)

    # Load model
    model = CryptoGNN(num_features=8, num_classes=3)
    model.load_state_dict(torch.load("crypto_advanced.pt", map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(graph.x, graph.edge_index, graph.batch)
        softmax = F.softmax(output, dim=1)
        confidence, prediction = torch.max(softmax, dim=1)

    # Class labels
    classes = ["AES-128", "SHA-256", "RSA"]

    detected_algo = classes[prediction.item()]
    confidence_score = confidence.item() * 100

    # File name without .txt or .py
    function_name = os.path.basename(file_path)
    function_name = function_name.replace(".py", "").replace(".txt", "")

    # Print final result
    print_result(function_name, detected_algo, confidence_score)


if __name__ == "__main__":
    main()
