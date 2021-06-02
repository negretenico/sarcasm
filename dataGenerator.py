
from __future__ import unicode_literals, print_function, division
import unicodedata
import pandas as pd 
import re
import io
import pickle
import os
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
class Data:
    def __init__(self) -> None:
        self.PATH = os.getcwd()
        self.df = pd.read_csv(self.PATH+"//Data//train-balanced-sarcasm.csv")
        self.labels = self.df["label"]
        self.LIMIT = 50000
        self.wordDict = {}
        self.idCounter =0
        self.MAX_LENGTH = 15
        self.eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

    def get_not_sarcastic_comments(self):
        return " ".join(self.df[self.df["label"]==0]["comment"].fillna(""))
    
    def get_sarcastic_comments(self):
        return " ".join(self.df[self.df["label"]==1]["comment"].fillna(""))


    def vectorize_input(self,sentence):
        vector = np.zeros(self.idCounter)
        allWords = sentence.split(" ")
        print("---------------------------Vectorizing Current Input---------------------------")
        for i, word in enumerate(allWords):
            if word in self.wordDict:
                vector[self.wordDict[word]] = 1
        return vector

    # Lowercase, trim, and remove non-letter characters
    def export_to_txt(self):
        df= self.df[["parent_comment","comment"]]
        array = np.array(df)
        with open("Data/comments_responses.txt","w") as f:
            for row in array:
                try:
                    f.write("\t".join(row)) 
                    f.write("\n")
                except:
                    print("File writing error")  


    def get_train_test_data(self):
        X,y = self.generate_feature()
        xTrain,xTest,yTrain,yTest =train_test_split(X, y, test_size = 0.2, random_state = 0)
        return xTrain,yTrain,xTest,yTest
    
    def unicodeToAscii(self,s):
      return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
    def filterPair(self,p):
        return len(p)==2 and len(p[0].split(' ')) < self.MAX_LENGTH and \
            len(p[1].split(' ')) < self.MAX_LENGTH and \
            p[1].startswith(self.eng_prefixes)

    def prepareData(self,comment, response, reverse=False):
        input_comment, output_comment, pairs = self.readFile(comment, response, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_comment.addSentence(pair[0])
            output_comment.addSentence(pair[1])
        print("Counted words:")
        print(input_comment.name, input_comment.n_words)
        print(output_comment.name, output_comment.n_words)
        return input_comment, output_comment, pairs

    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def readFile(self,comment, response, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('Data/%s_%s.txt' % (comment, response), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Comment(response)
            output_lang = Comment(comment)
        else:
            input_lang = Comment(response)
            output_lang = Comment(comment)

        return input_lang, output_lang, pairs
class Comment:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1