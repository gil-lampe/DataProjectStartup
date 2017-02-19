#!/usr/bin/env python

"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017

Community detection
"""

import os
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn import cluster

# you need to be in the directory where 'community_detection.py' is located
# to be able to import
# you can use os.chdir(path)
from community_detection import louvain


# Load the graph into an undirected NetworkX graph

##################
# your code here #
##################

# hint: use read_edgelist function of NetworkX
G = nx.read_edgelist("../dataset/CA-hepth.txt")



# Get giant connected component (GCC)

GCC = max(nx.connected_component_subgraphs(G),key = len)

# hint: use connected_component_subgraphs function of NetworkX, GCC is the biggest of the subgraphs




# Spectral clustering algorithm
# Implement and apply spectral clustering

def spectral_clustering(G, k):
    L = nx.normalized_laplacian_matrix(G).astype(float) # Normalized Laplacian
    A = nx.adjacency_matrix(G)

    # Calculate k smallest in magnitude eigenvalues and corresponding eigenvectors of L
    eigval,eigvec = eigs(L,K=k,which="SR")

    # hint: use eigs function of scipy

    eigval = eigval.real # Keep the real part
    eigvec = eigvec.real # Keep the real part
    # sort is implemented by default in increasing order
    idx = eigval.argsort() # Get indices of sorted eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    
    # Perform k-means clustering (store in variable "membership" the clusters to which points belong)
    # Initialize the cluster randomly
    
    # hint: use KMeans class of scikit-learn


    membership = list(km.labels_)
    # will contain node IDs as keys and membership as values
    clustering = {}
    nodes = G.nodes()
    for i in range(len(nodes)):
        clustering[nodes[i]] = membership[i]
    
    return clustering
	
# Apply spectral clustering to the CA-HepTh dataset
clustering = spectral_clustering(G=GCC, k=60)

# sanity check
GCC.number_of_nodes() == len(clustering)

# Modularity
# Implement and compute it for two clustering results

# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    n_clusters = len(list(set(clustering.values())))
    modularity = 0 # Initialize total modularity value
    #Iterate over all clusters
    for i in range(n_clusters):
        # Get the nodes that belong to the i-th cluster
        nodeList = [n for n,v in clustering.iteritems() if v == i]
        
        # Create the subgraphs that correspond to each cluster and compute modularity as in equation 1

        ##################
        # your code here #
        ##################
    
        # hint: use subgraph(nodeList) function to get subgraph induced by nodes in nodeList			
        
    return modularity
	
print "Modularity Spectral Clustering: ", modularity(GCC, clustering)

# Implement random clustering
k = 60
r_clustering = {}

# Partition randomly the nodes in k clusters (store the clustering result in the r_clustering dictionary)

##################
# your code here #
##################
    
# hint: use randint function



	
print "Modularity Random Clustering: ", modularity(GCC, r_clustering)

# Louvain
# Run it and compute modularity

# Partition graph using the Louvain method
clustering = louvain(GCC)

print "Modularity Louvain: ", modularity(GCC, clustering)
