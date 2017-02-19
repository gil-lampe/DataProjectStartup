import re
import xlsxwriter
import sys
import word2vec
import nltk
#nltk.download()
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from os import listdir

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('techcrunch.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"name of file")
worksheet.write(0,1,"number of funding")
worksheet.write(0,2,"ventures")
worksheet.write(0,3,"series")
worksheet.write(0,4,"seed")
worksheet.write(0,5,"possibility of investment")

# tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#function

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    #review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
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

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

nb_of_file = 0

#print(onlyfiles)
for f in listdir('/Users/chen/Desktop/DataProject/techcrunch data/'):
    nb_of_file = nb_of_file + 1
    worksheet.write(nb_of_file, 0, f)
    file = open(f,"r")
    review = file.read()
    # print(review)
    wordlist = review_to_wordlist(review)
    l = len(wordlist)
    compteur = 0
    ligne = 0

    for i in range(l):
        million = re.search("million", wordlist[i])
        billion = re.search("billion", wordlist[i])
        venture = re.search("ventures", wordlist[i])
        serie = re.search("serie", wordlist[i])
        seed = re.search("seed", wordlist[i])
        # print(a)
        if million is not None:
            compteur = compteur + 1
            #print(wordlist[i])
            worksheet.write(nb_of_file, 1, str(wordlist[i - 1] + " " + wordlist[i]))

        if billion is not None:
            compteur = compteur + 1
            worksheet.write(nb_of_file, 1, str(wordlist[i - 1] + " " + wordlist[i]))

        if venture is not None:
            compteur = compteur + 1
            #print(i)
            worksheet.write(nb_of_file, 2, str(wordlist[i - 3] + " " + wordlist[i - 2] + " " + wordlist[i - 1] + " " + wordlist[i]))

        if serie is not None:
            if i+1 < l:
                worksheet.write(nb_of_file, 3, wordlist[i] + " " + wordlist[i + 1])
            else :
                worksheet.write(nb_of_file, 3, wordlist[i])
        if seed is not None:
            #print(seed.group(0))
            #print(i ,l)
            compteur = compteur + 1
            if i+2 < l :
                worksheet.write(nb_of_file, 3, str(wordlist[i - 2] + " " + wordlist[i - 1] + " " + wordlist[i] + " " + wordlist[i + 1] + " " + wordlist[i + 2]))
            else:
                worksheet.write(nb_of_file, 3, str(wordlist[i - 1] + " " + wordlist[i]))

    if compteur > 2:
        worksheet.write(nb_of_file, 5, "yes")
    else:
        worksheet.write(nb_of_file, 5, "no")

'''
# argv is your commandline arguments, argv[0] is your program name, so skip it
for n in sys.argv[1:]:
    print(n) #print out the filename we are currently processing
    input = open(n, "r")
    output = open(n + ".out", "w")
    print(n.name)
    # do some processing
    input.close()
    output.close()

sentences = []  # Initialize an empty list of sentences

file = open("%e2%80%8bnew-app-%e2%80%8btagly-%e2%80%8bbets-on-%e2%80%8bconnect%e2%80%8bing%e2%80%8b-consumers-with-brand-content.txt","r")
review = file.read()
#print(review)

wordlist = review_to_wordlist(review)
#sentences = review_to_sentences(review,tokenizer)

#print("wordlist")
#print(wordlist)
#print(type(wordlist))
#print("sentences")
#print(sentences)
l = len(wordlist)
compteur = 0
ligne = 0

for i in range(l) :
    million = re.search("million",wordlist[i])
    venture = re.search("ventures", wordlist[i])
    serie = re.search("serie", wordlist[i])
    seed = re.search("seed", wordlist[i])
    #print(a)
    if million is not None:
        print(million.group(0))
        print(wordlist[i-1])
        print(type(wordlist[i - 1]))
        compteur = compteur + 1

        worksheet.write(1, 1, str(wordlist[i - 1]))

    if venture is not None:
        compteur = compteur + 1
        print(venture.group(0))
        print(str(wordlist[i-3] + " " + wordlist[i-2] + " "+ wordlist[i-1] + " " + wordlist[i]))

        worksheet.write(1, 2, str(wordlist[i-3] + " " + wordlist[i-2] + " "+ wordlist[i-1] + " " + wordlist[i]))


    if serie is not None:
        print(serie.group(0))
        print(str(wordlist[i] + " " + wordlist[i+1] ))

        worksheet.write(1, 3, wordlist[i] + " " + wordlist[i + 1])

    if seed is not None:
        print(seed.group(0))
        print(str(wordlist[i-2] + " " + wordlist[i-1] + " " + wordlist[i]+ " " + wordlist[i+1]+ " " + wordlist[i+2]))

        worksheet.write(1, 3, str(wordlist[i-2] + " " + wordlist[i-1] + " " + wordlist[i]+ " " + wordlist[i+1]+ " " + wordlist[i+2]))

if compteur > 2 :
    print("investment")
    worksheet.write(1, 5, "yes")
else :
    worksheet.write(1,5,"no") '''


