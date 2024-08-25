import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors
import numpy as np

def load_and_analyze_graph(graph_file):
    # Load the graph
    G = nx.read_graphml(graph_file)
    
    print(f"Graph loaded from {graph_file}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Basic graph statistics
    print("\nGraph Statistics:")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    print("\nDegree Distribution:")
    for degree, count in sorted(degree_counts.items())[:10]:  # Print top 10
        print(f"Degree {degree}: {count} nodes")
    
    # Most common relations
    relations = [data['relation'] for u, v, data in G.edges(data=True)]
    relation_counts = Counter(relations)
    print("\nMost Common Relations:")
    for relation, count in relation_counts.most_common(10):  # Print top 10
        print(f"{relation}: {count}")
    
    """# Print all node names
    print("\nAll Node Names:")
    for node in G.nodes():
        print(G.nodes[node]['name'])"""
    
    # Visualize the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=20, node_color='lightblue', 
            with_labels=False, edge_color='gray', alpha=0.6)
    
    # Add labels to some nodes (e.g., top 10 by degree)
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    labels = {node: G.nodes[node]['name'] for node, degree in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches='tight')
    print("\nGraph visualization saved as 'graph_visualization.png'")

    # Show the plot (optional, comment out if running on a server without display)
    plt.show()


def visualize_connected_graph(graph_file):
    # Load the graph
    G = nx.read_graphml(graph_file)
    
    print(f"Graph loaded from {graph_file}")
    print(f"Original graph - Number of nodes: {G.number_of_nodes()}")
    print(f"Original graph - Number of edges: {G.number_of_edges()}")
    
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc).copy()
    
    print(f"\nLargest connected component - Number of nodes: {G_connected.number_of_nodes()}")
    print(f"Largest connected component - Number of edges: {G_connected.number_of_edges()}")
    
    # Basic graph statistics for the connected component
    print("\nConnected Graph Statistics:")
    print(f"Is connected: {nx.is_connected(G_connected)}")
    
    # Degree distribution
    degrees = [d for n, d in G_connected.degree()]
    degree_counts = Counter(degrees)
    print("\nDegree Distribution:")
    for degree, count in sorted(degree_counts.items())[:10]:  # Print top 10
        print(f"Degree {degree}: {count} nodes")
    
    # Most common relations
    relations = [data['relation'] for u, v, data in G_connected.edges(data=True)]
    relation_counts = Counter(relations)
    print("\nMost Common Relations:")
    for relation, count in relation_counts.most_common(10):  # Print top 10
        print(f"{relation}: {count}")
    
    # Visualize the connected graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_connected)
    nx.draw(G_connected, pos, node_size=20, node_color='lightblue', 
            with_labels=False, edge_color='gray', alpha=0.6)
    
    # Add labels to some nodes (e.g., top 10 by degree)
    top_nodes = sorted(G_connected.degree, key=lambda x: x[1], reverse=True)[:30]
    labels = {node: G_connected.nodes[node]['name'] for node, degree in top_nodes}
    nx.draw_networkx_labels(G_connected, pos, labels, font_size=8)
    
    plt.title("Connected Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("connected_graph_visualization.png", dpi=300, bbox_inches='tight')
    print("\nConnected graph visualization saved as 'connected_graph_visualization.png'")

    # Show the plot (optional, comment out if running on a server without display)
    plt.show()


def visualize_coloured_graph(graph_file):
    # Load the graph
    G = nx.read_graphml(graph_file)
    
    print(f"Graph loaded from {graph_file}")
    print(f"Original graph - Number of nodes: {G.number_of_nodes()}")
    print(f"Original graph - Number of edges: {G.number_of_edges()}")
    
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc).copy()
    
    print(f"\nLargest connected component - Number of nodes: {G_connected.number_of_nodes()}")
    print(f"Largest connected component - Number of edges: {G_connected.number_of_edges()}")
    
    # Degree distribution
    degrees = dict(G_connected.degree())
    degree_values = list(degrees.values())
    degree_counts = Counter(degree_values)
    print("\nDegree Distribution:")
    for degree, count in sorted(degree_counts.items())[:10]:  # Print top 10
        print(f"Degree {degree}: {count} nodes")
    
    # Visualize the connected graph
    fig, ax = plt.subplots(figsize=(15, 12))
    pos = nx.spring_layout(G_connected, k=0.5, iterations=50)
    
    # Color nodes based on their degree using a log scale
    node_colors = list(degrees.values())
    vmin = max(1, min(node_colors))  # Avoid log(0)
    vmax = max(node_colors)
    
    # Create a custom colormap with a logarithmic normalization
    cmap = plt.cm.viridis
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    
    # Apply the color mapping manually
    node_colors = [cmap(norm(degree)) for degree in node_colors]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G_connected, pos, ax=ax, node_size=30, 
                                   node_color=node_colors)
    
    # Draw edges
    nx.draw_networkx_edges(G_connected, pos, ax=ax, alpha=0.1, width=0.1)
    
    # Add labels to more nodes (e.g., top 30 by degree)
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:30]
    labels = {node: G_connected.nodes[node]['name'] for node in top_nodes}
    nx.draw_networkx_labels(G_connected, pos, labels, ax=ax, font_size=8, font_weight='bold')
    
    ax.set_title("Connected Graph Visualization (Log Scale)", fontsize=16)
    ax.axis('off')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Node Degree (Log Scale)", pad=0.05)
    
    plt.tight_layout()
    plt.savefig("connected_graph_visualization_colored_log.png", dpi=300, bbox_inches='tight')
    print("\nConnected graph visualization saved as 'connected_graph_visualization_colored_log.png'")

    # Show the plot (optional, comment out if running on a server without display)
    plt.show()

if __name__ == "__main__":
    graph_file = './Datasets/FinalDB/NGKG.graphml'
    #load_and_analyze_graph(graph_file)
    visualize_coloured_graph(graph_file)