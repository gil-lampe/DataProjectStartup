import string
import re 
import itertools
import copy
import igraph
import nltk
import operator
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag

# might also be required:
# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('stopwords')

# import custom functions
from library import clean_text_simple, terms_to_graph, unweighted_k_core

my_doc = '''Waiting for the wave to crest [wavelength services]
Wavelength services have been hyped ad nauseam for years. But despite their
quick turn-up time and impressive margins, such services have yet to
live up to the industry's expectations. The reasons for this lukewarm
reception are many, not the least of which is the confusion that still
surrounds the technology, but most industry observers are still
convinced that wavelength services with ultimately flourish'''

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc)
                              
g = terms_to_graph(my_tokens, w=4)
    
# number of edges
len(g.es)

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))
print(g.vs)

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

edge_weights

for w in range(2,11):
    g = terms_to_graph(my_tokens, w)
    print g.density()
    
# decompose g
g = terms_to_graph(my_tokens, w=4)
core_numbers = unweighted_k_core(g)

# compare with igraph method
dict(zip(g.vs["name"],g.coreness()))

# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]

print(max_c_n, keywords, len(keywords))