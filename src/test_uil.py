import torch
from torch_geometric.data import Data, Batch
from models.UIL import UILModel

def generate_random_graph(num_nodes, num_edges, num_node_features):
    x = torch.randn((num_nodes, num_node_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)

def create_dummy_batch(batch_size, num_node_features, num_classes):
    graphs = []
    for _ in range(batch_size):
        num_nodes = torch.randint(5, 10, (1,)).item()
        num_edges = torch.randint(10, 20, (1,)).item()
        graph = generate_random_graph(num_nodes, num_edges, num_node_features)
        graph.y = torch.randint(0, num_classes, (1,))
        graphs.append(graph)
    return Batch.from_data_list(graphs)

def test_uil_forward():
    # Mock config
    config = {
        'hidden_dim': 32,
        'num_layers': 2,
        'dropout': 0.1,
        'pooling': 'mean',  # unused in encoder, included for completeness
        'weight_ce': 1.0,
        'weight_str': 0.0,  # skip structural loss
        'weight_sem': 1.0,
        'weight_reg': 1.0
    }

    dataset_info = {
        'num_features': 8,
        'num_classes': 3
    }

    model = UILModel(config, dataset_info, rho=0.5)
    model.eval()  # inference mode to disable dropout

    for i in range(2):
        batch = create_dummy_batch(batch_size=2, num_node_features=8, num_classes=3)
        output = model(batch, target=batch.y)

        print(f"\nBatch {i+1} Output Summary:")
        print(f"Logits shape: {output['logits'].shape}")
        print(f"Predicted labels: {output['logits'].argmax(dim=-1)}")
        print(f"True labels: {batch.y}")
        print(f"Loss total: {output['loss_total'].item():.4f}")
        print(f"CE Loss: {output['loss_ce'].item():.4f}")
        print(f"Semantic Loss: {output['loss_sem'].item():.4f}")
        print(f"Reg Loss: {output['loss_reg'].item():.4f}")
        print(f"Node mask avg: {output['node_mask'].mean().item():.4f}")
        print(f"Edge mask avg: {output['edge_mask'].mean().item():.4f}")
        if i == 0:
            print("\nOriginal graph and predicted stable subgraph (graph 0 in batch):")

            # Select graph 0's nodes and edges
            node_mask = output['node_mask'].squeeze()
            edge_mask = output['edge_mask'].squeeze()
            edge_index = output['cached_masks']['edge_index']
            batch_vec = output['cached_masks']['batch']  # maps node to graph index

            graph0_node_idx = (batch_vec == 0).nonzero(as_tuple=True)[0]
            print(f"Graph 0 original node indices: {graph0_node_idx.tolist()}")

            # Determine which nodes are kept (threshold at 0.5)
            node_mask_bin = (node_mask >= 0.5).float()
            kept_nodes = graph0_node_idx[node_mask_bin[graph0_node_idx] >= 0.5]
            print(f"Graph 0 predicted stable nodes (mask ≥ 0.5): {kept_nodes.tolist()}")

            # Determine which edges belong to graph 0 (both endpoints in graph0)
            graph0_node_set = set(graph0_node_idx.tolist())
            edge_in_graph0 = [
                i for i, (src, dst) in enumerate(edge_index.t().tolist())
                if src in graph0_node_set and dst in graph0_node_set
            ]
            kept_edges = [
                i for i in edge_in_graph0 if edge_mask[i] >= 0.3
            ]

            print(f"Total edges in graph 0: {len(edge_in_graph0)}")
            print(f"Stable edges predicted (mask ≥ 0.3):")
            for i in kept_edges:
                src, dst = edge_index[:, i].tolist()
                print(f"  edge ({src} → {dst}), mask = {edge_mask[i].item():.2f}")


if __name__ == "__main__":
    test_uil_forward()
