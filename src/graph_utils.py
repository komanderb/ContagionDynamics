import networkx as nx
from scipy.sparse.linalg import matrix_power
import numpy as np
#TODO: different models to generate random graphs


def generate_twitter_like_graph(num_nodes, avg_out_degree):
    # Generate an undirected Barabási-Albert graph
    m = avg_out_degree // 2  # Number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(num_nodes, m)

    # Convert to directed graph by assigning direction to each edge randomly
    DG = nx.DiGraph()
    for u, v in G.edges():
        if np.random.rand() > 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)

    return DG


def second_degree_neighbor(graph, directed=True):
    """
    A function to connect all nodes to their second degree neighbors through squaring the adjancecy matrix.
    Potentially better approach could be algorithmically (BFS)
    :param graph:
    :return:
    """

    adj_matrix  = nx.to_scipy_sparse_array(graph)
    square_adj_matrix = matrix_power(adj_matrix , power=2)
    # remove selfloops
    square_adj_matrix.setdiag(0)
    # unweighted version, otherwise different considerations??
    square_adj_matrix[square_adj_matrix > 0] = 1
    if directed:
        graph_ext = nx.from_scipy_sparse_array(square_adj_matrix, create_using=nx.DiGraph)
    else:
        graph_ext = nx.from_scipy_sparse_array(square_adj_matrix)

    return graph_ext

