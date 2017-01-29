import numpy
import xlsxwriter
import sys
import word2vec
import nltk
nltk.download()
# Create a workbook and add a worksheet.
#workbook = xlsxwriter.Workbook('techcrunch.xlsx')
#worksheet = workbook.add_worksheet()

# open data
tokenizer = nltk.data.load("3-gadgets.txt")

print(tokenizer)

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
#for item, cost in (a):
 #   worksheet.write(row, col,     item)
  #  worksheet.write(row, col + 1, cost)
   # row += 1

#workbook.close()