#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017
"""

# Import modules
from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.sparse as sparse
from numpy import *
import numpy.linalg



############## Question 1
# Load the graph into an undirected NetworkX graph

##################
# your code here #
##################

# hint: use read_edgelist function of NetworkX
G = nx.read_edgelist("../dataset/CA-hepth.txt")




############## Question 2
# Network Characteristics
print 'Number of nodes:', G.number_of_nodes() 
print 'Number of edges:', G.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G)




# Get giant connected component (GCC)

GCC = max(nx.connected_component_subgraphs(G),key = len)
#print(GCC)

# hint: use connected_component_subgraphs function of NetworkX, GCC is the biggest of the subgraphs





# Compute the fraction of nodes and edges in GCC 

print 'Fraction of nodes in GCC', GCC.number_of_nodes()/G.number_of_nodes ()
print 'Fraction of nodes in GCC', GCC.number_of_edges()/G.number_of_edges ()




############## Question 3
# Extract degree sequence and compute min, max, median and mean degree

degree_sequence = G.degree().values()
#print 'degree sequence:', degree_sequence

max = np.max(degree_sequence)
min = np.min(degree_sequence)
average = np.average(degree_sequence)
median = np.median(degree_sequence)
mean = np.mean(degree_sequence)
# hint: use the min, max, median and mean functions of NumPy


# Degree distribution
y=nx.degree_histogram(G)
plt.figure(1)
plt.plot(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#f.savefig("degree.png",format="png")

plt.figure(2)
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#s.savefig("degree_loglog.png",format="png")
