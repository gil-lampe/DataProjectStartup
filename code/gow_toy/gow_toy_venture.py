import os
import random
import string
import re
import itertools
import copy
import igraph
import nltk
import operator
from nltk.corpus import stopwords
import xlsxwriter

# requires nltk 3.2.1
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# might also be required:
nltk.download('maxent_treebank_pos_tagger')
nltk.download('stopwords')

from library import clean_text_simple, terms_to_graph, accuracy_metrics

stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')

#build new workbook to stock key words
workbook = xlsxwriter.Workbook('Keywords_data.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"name of file")
worksheet.write(0,1,"key words")


##################################
# read and pre-process articles #
##################################

# to fill
path_to_articles = '/Users/chen/Desktop/DataProject/techcrunch data/'
article_names = sorted(os.listdir(path_to_articles))

articles = []
counter = 0

for filename in article_names:
    # read file
    worksheet.write(counter + 1,0, filename)
    with open(path_to_articles + filename, 'r') as my_file:
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text = re.sub('\s+', ' ', text)
    articles.append(text)

    counter += 1

    if counter % 100 == 0:
        print counter, 'files processed'

articles_cleaned = []
counter = 0

for article in articles:
    my_tokens = clean_text_simple(article)
    articles_cleaned.append(my_tokens)
    counter += 1
    if counter % 100 == 0:
        print counter, 'articles processed'


keywords_gow = []
counter = 0

for article in articles_cleaned:
    #register article's name

    # create graph-of-words
    g = terms_to_graph(article, w=4)
    # decompose graph-of-words
    core_numbers = dict(zip(g.vs['name'], g.coreness()))
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
    l = len(keywords)
    for i in range(l):
        worksheet.write(counter + 1, i, keywords[i])
    # save results
    keywords_gow.append(keywords)

    counter += 1
    if counter % 100 == 0:
        print counter, 'articles processed'


