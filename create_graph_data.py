import os
import json
import random

NUM_GRAPHS = 500
NODE_FEATURES = 8
MIN_NODES = 5
MAX_NODES = 20

def generate_graph():
    num_nodes = random.randint(MIN_NODES, MAX_NODES)

    # node_features → loader expects this key
    node_features = []
    for _ in range(num_nodes):
        node_features.append([round(random.random(), 4) for _ in range(NODE_FEATURES)])

    # edges → loader expects this key
    edges = []
    for _ in range(num_nodes * 2):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src != dst:
            edges.append([src, dst])

    # label → loader expects this key
    label = random.randint(0, 2)

    return {
        "node_features": node_features,
        "edges": edges,
        "label": label
    }

def main():
    save_path = "data/raw"
    os.makedirs(save_path, exist_ok=True)

    graphs = []
    print(f"Generating {NUM_GRAPHS} graphs...")

    for _ in range(NUM_GRAPHS):
        graphs.append(generate_graph())

    json_path = os.path.join(save_path, "graphs.json")

    with open(json_path, "w") as f:
        json.dump(graphs, f, indent=2)

    print("\nDataset Created Successfully!")
    print(f"Path: {json_path}")
    print(f"Total Graphs: {NUM_GRAPHS}")

if __name__ == "__main__":
    main()
