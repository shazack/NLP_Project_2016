
import nltk
import math 
import os
import sys


def bigrams_pr(reviews):
bigram_prob = {}

for sentence in reviews:
    token = sentence.split()
    token = [START_SYMBOL] + tokens #Add a start symbol 
    #so the first word would count as bigram
    bigrams = (tuple(nltk.bigrams(token)))
    for bigram in bigrams:
        if bigram not in bigram_p:
           bigram_prob[bigram] = 1
        else:
           bigram_prob[bigram] += 1

        for bigram in bigram_p:
            if bigram[0] == '*':  
                bigram_prob[bigram] = math.log(bigram_p[bigram]/unigram_prob[('STOP',)],2)
            else:
                bigram_prob[bigram] = math.log(bigram_p[bigram]/unigram_prob[(word[0],)],2)
                
                


  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i], input_list[i+1]))
  return bigram_list


print(find_bigrams(input_list))

file= sys.argv[1]

bigrams_pr(file)
