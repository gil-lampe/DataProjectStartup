from __future__ import print_function
import re
from bs4 import BeautifulSoup
import xlsxwriter
from nltk.corpus import stopwords
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

directory = '/Users/chen/Desktop/DataProject/techcrunch data/'
workbook = xlsxwriter.Workbook('Train_data.xlsx')
worksheet = workbook.add_worksheet()
reverse_dictionary = xlsxwriter.Workbook('reverse_dictionary.xlsx')
worksheet_reverse_dictionary = reverse_dictionary.add_worksheet()
batch_labels = xlsxwriter.Workbook('reverse_dictionary.xlsx')
worksheet_batch_labels = batch_labels.add_worksheet()
vocabulary_size = 50000
wordlist=[]
data_index = 0

#train the skip-gram model with fellow number

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
num_steps = 100001

graph = tf.Graph()

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review).get_text()

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  compteur = 0
  for word in words:
    compteur = compteur + 1
    if word in dictionary:
      index = dictionary[word]
      worksheet_reverse_dictionary.write(2, compteur, dictionary[word])
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
      worksheet_reverse_dictionary.write(2, compteur, unk_count)
    data.append(index)
    worksheet_reverse_dictionary.write(0, compteur, index)
    worksheet_reverse_dictionary.write(0, compteur, words)

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
      worksheet_batch_labels.write(1,i * num_skips + j, batch)
      worksheet_batch_labels.write(2, i * num_skips + j, labels)
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels


nb_of_file = 0

for f in os.listdir('/Users/chen/Desktop/DataProject/techcrunch data/'):

    nb_of_file = nb_of_file + 1
    #print(f)
    file = open('/Users/chen/Desktop/DataProject/techcrunch data/'+ f,"r")
    review = file.read()
        #print(review_to_wordlist(review))
    worksheet.write(1, nb_of_file, nb_of_file)
    worksheet.write(2, nb_of_file, review)
    wordlist = wordlist +  review_to_wordlist(review)
        #print(wordlist)


#print(wordlist)

print(len(wordlist))
#print(type(wordlist), wordlist[990:1000])


data, count, dictionary, reverse_dictionary = build_dataset(wordlist)
del wordlist
print('Most common words (+UNK)', count[:20])
print('Sample data', data[:10])
print('reverse dictionary',reverse_dictionary)
print(len(reverse_dictionary))


for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      #print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
            if nearest[k]<= len(reverse_dictionary):
                close_word = reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
            else:
                pass
        print(log)
  final_embeddings = normalized_embeddings.eval()

  num_points = 400

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])


  def plot(embeddings, labels):
      assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
      pylab.figure(figsize=(15, 15))  # in inches
      for i, label in enumerate(labels):
          x, y = embeddings[i, :]
          pylab.scatter(x, y)
          pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                         ha='right', va='bottom')
      pylab.show()


  words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
  plot(two_d_embeddings, words)
