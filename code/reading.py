from os import listdir

for f in listdir('/Users/chen/Desktop/DataProject/techcrunch data/'):
    print(f)
    file = open('/Users/chen/Desktop/DataProject/techcrunch data/' + f, "r")

    review = file.read()
    print(review)
    raw_input("Next, push your escape")