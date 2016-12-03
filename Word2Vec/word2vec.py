import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

def getAvgFeatureVecs(reviews, model, numberOfFeatures):
    reviewFeatureVecs = np.zeros((len(reviews),numberOfFeatures),dtype="float32")
    cnt=0
    for review in reviews:
       reviewFeatureVecs[cnt] = makeFeatureVec(review, model, numberOfFeatures)
       cnt+=1
    return reviewFeatureVecs

def makeFeatureVec(words, model, numberOfFeatures):
    featureVec = np.zeros((numberOfFeatures,),dtype="float32")
    nwords = 0.
    wordSet = set(model.index2word)
    for word in words:
        if word in wordSet:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    sentence = tokenizer.tokenize(review.strip())
    final = []
    for sntc in sentence:
        if len(sntc) > 0:
            final.append( review_to_wordlist( sntc, remove_stopwords ))
    return final

def review_to_wordlist( review, remove_stopwords=False ):
    rvw = BeautifulSoup(review,"html.parser").get_text()
    rvw = re.sub("[^a-zA-Z]"," ", rvw)
    words = rvw.lower().split()
    if remove_stopwords:
        stopwrds = set(stopwords.words("english"))
        words = [w for w in words if not w in stopwrds]
    return(words)

def getCleanReviews(reviews):
    clean = []
    for review in reviews["review"]:
        clean.append( review_to_wordlist( review, remove_stopwords=True ))
    return clean

if __name__ == '__main__':
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    test=train[18750:25000]
    train=train[:18750]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = [] 
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    model = Word2Vec(sentences, workers=4, size=300, min_count = 40, window = 10, sample = 0.0001, seed=1)
    model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, 300 )
    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, 300 )
    forest = RandomForestClassifier( n_estimators = 10 )
    forest = forest.fit( trainDataVecs, train["sentiment"] )
    result = forest.predict( testDataVecs )
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Word2VecOutput.csv", index=False, quoting=3 )
    print("Wrote Word2VecOutput.csv")

    actual=test["sentiment"]
    predicted=output["sentiment"]
    true=0.0
    total=0.0
    for i in range(18750,25000):
        if(actual[i]==predicted[i]):
            true+=1.0
        total+=1.0
    print("Accuracy is "+str(true/total))