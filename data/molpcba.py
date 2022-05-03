import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib

from tqdm.std import tqdm


def laplace_decomp(graph, max_freqs, device):
    g: dgl.DGLGraph
    g, label = graph

    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N


    EigVals, EigVecs = np.linalg.eig(L.toarray())
    idx = EigVals.argsort()[0 : max_freqs] # Keep up to the maximum desired number of frequencies
    EigVals, EigVecs = EigVals[idx], np.real(EigVecs[:,idx])

    #Sort, normalize and pad EigenVectors
    EigVecs = EigVecs[:, EigVals.argsort()]# increasing order

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs - n), value=float('nan')).to(device)
    else:
        g.ndata['EigVecs'] = EigVecs.to(device)

    # Save eigenvalues and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(
        EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    # Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(), 1).unsqueeze(2).to(device)

    return g, label


def make_full_graph(graph):
    g, label = graph

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Copy over the node feature data and laplace eigvals/eigvecs
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Initalize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over, and identify real edges!
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)  # This indicates real edges

    return full_g, label


def add_edge_laplace_feats(graph):
    g, label = graph

    EigVals = g.ndata['EigVals'][0].flatten()

    source, dest = g.find_edges(g.edges(form='eid'))

    # Compute diffusion differences and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs'] - g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(), 1).unsqueeze(2)

    # No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']

    return g, label