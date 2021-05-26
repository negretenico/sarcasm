import enum
from numpy.lib.function_base import vectorize
import pandas as pd 
import numpy as np 
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import mitdeeplearning as mdl
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

class Data:
    def __init__(self) -> None:
        self.PATH = os.getcwd()
        self.df = pd.read_csv(self.PATH+"//Data//train-balanced-sarcasm.csv")
        self.labels = self.df["label"]
        self.LIMIT = 50000
        self.wordDict = {}
        self.idCounter =0

    def get_not_sarcastic_comments(self):
        return " ".join(self.df[self.df["label"]==0]["comment"].fillna(""))
    
    def get_sarcastic_comments(self):
        return " ".join(self.df[self.df["label"]==1]["comment"].fillna(""))

    def tfId_features(self):
        tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2))
        feature= tfidf.fit_transform(np.array((self.df['comment'].fillna(""))))
        with open("tfIDFeatures.pickle", "wb") as f:
            pickle.dump(feature, f)

    def vectorize_input(self,sentence):
        vector = np.zeros(self.idCounter)
        allWords = sentence.split(" ")
        print("---------------------------Vectorizing Current Input---------------------------")
        for i, word in enumerate(allWords):
            if word in self.wordDict:
                vector[self.wordDict[word]] = 1
        return vector

    def generate_feature(self):
        print("---------------------------Creating Features---------------------------")
        self.df["comment"].fillna("",inplace =True)
        localWordDict = {}
        localIdCounter = 0
        for i in range(self.LIMIT):
            allWords = self.df.iloc[i,1].split(" ")

            for word in allWords:
                if word not in localWordDict:
                    localWordDict[word] = localIdCounter
                    localIdCounter += 1
        X = np.zeros((self.LIMIT, localIdCounter),dtype='float')
        y = np.array(self.df.iloc[0:self.LIMIT,0])
        self.wordDict = localWordDict
        self.idCounter = localIdCounter
        for i in range(self.LIMIT):

            allWords = self.df.iloc[i,1].split(" ")
            for word in allWords:
                X[i, self.wordDict[word]]  = 1
        return X,y

    def generate_vocab(self):
        comments = "".join(self.data.df['comment'].fillna(""))
        vocab = sorted(set(comments))
        with open("vocab.pickle", "wb") as f:
            pickle.dump(vocab, f)       

    def get_train_test_data(self):
        X,y = self.generate_feature()
        xTrain,xTest,yTrain,yTest =train_test_split(X, y, test_size = 0.2, random_state = 0)
        return xTrain,yTrain,xTest,yTest

class Preceptron:
    def __init__(self,learning_rate,num_epoches,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------Preceptron---------------------------")
        self.learning_rate = learning_rate
        self.num_epoches = num_epoches
        self.xTrain,self.yTrain, self.xTest,self.yTest = trainFeature,trainLabel,testFeature,testLabel
        self.weights = np.random.rand(np.shape(self.xTrain)[1]+1)

    def step(self,x):
      return (np.sign(x)+1)/2

    def plr(self,weights,example,target):
        bias = 1
        example=np.hstack((example,bias)) #Add bias as extra "input"
        o=self.step(np.dot(example,weights))
        # compute weight update
        e=target-o
        new_weights=weights+self.learning_rate*e*example
        # return updated weights
        return new_weights
    
    def predict(self,feature):
        predictions =[]
        for example in feature:
            predictions.append(self.step(np.dot(np.hstack((example,1)),self.weights)))
        return np.array(predictions)
    
    def run_epoch(self,weights,data,targets):
        nexamples=data.shape[0]
        order=np.random.permutation(nexamples)
        for i in range(nexamples):
            weights=self.plr(weights,data[order[i]],targets[order[i]])
        return weights

    def score(self):
        prediciton =self.predict(self.xTest)
        return np.average(prediciton==self.yTest)

    def train(self):
        print("---------------------------Training has started---------------------------")
        old_weights= np.zeros(np.shape(self.weights))
        epoch = 0
        while (np.any(old_weights != self.weights )):
            old_weights=self.weights
            self.weights=self.run_epoch(self.weights,self.xTrain,self.yTrain)
            acc = self.score()
            if(epoch%10==0):
                print(f"Training on Epcoh {epoch}")
            if epoch >self.num_epoches:
                break
            epoch += 1
        with open("preceptron.pickle", "wb") as f:
            model = (self.weights,1)
            pickle.dump(model, f) 

class NaiveBaynes:
    def __init__(self,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------NaiveBaynes---------------------------")
        self.xTrain,self.yTrain, self.xTest,self.yTest = trainFeature,trainLabel,testFeature,testLabel
        self.probWordGivenPositive, self.probWordGivenNegative, self.priorPositive, self.priorNegative = self.compute_distros(self.xTrain,self.yTrain)
        min_prob = 1/self.yTrain.shape[0] #Assume very rare words only appeared once
        self.logProbWordPresentGivenPositive, self.logProbWordAbsentGivenPositive = self.compute_logdistros(self.probWordGivenPositive,min_prob)
        self.logProbWordPresentGivenNegative, self.logProbWordAbsentGivenNegative = self.compute_logdistros(self.probWordGivenNegative,min_prob)
        self.logPriorPositive, self.logPriorNegative = self.compute_logdistros(self.priorPositive,min_prob)
        with open("naiveBaynes.pickle", "wb") as f:
            model = (min_prob,   self.logProbWordPresentGivenPositive, self.logProbWordAbsentGivenPositive,self.logProbWordPresentGivenNegative, self.logProbWordAbsentGivenNegativeself.logPriorPositive, self.logPriorNegative )
            pickle.dump(model, f) 

    def compute_distros(self,x,y):
        print("---------------------------Computing Distributions---------------------------")
        # probWordGivenPositive: P(word|Sentiment = +ive)
        probWordGivenPositive=np.sum(x[y==0,:],axis=0) #Sum each word (column) to count how many times each word shows up (in positive examples)
        probWordGivenPositive=probWordGivenPositive/np.sum(y>=0) #Divide by total number of (positive) examples to give distribution

        # probWordGivenNegative: P(word|Sentiment = -ive)
        probWordGivenNegative=np.sum(x[y==1,:],axis=0)
        probWordGivenNegative=probWordGivenNegative/np.sum(y<0)

        # priorPositive: P(Sentiment = +ive)
        priorPositive = np.sum(y>=0)/y.shape[0] #Number of positive examples vs. all examples
        # priorNegative: P(Sentiment = -ive)
        priorNegative = 1 - priorPositive
        #  (note these last two form one distribution)

        return probWordGivenPositive, probWordGivenNegative, priorPositive, priorNegative
 
    def compute_logdistros(self,distros, min_prob):
        print("---------------------------Computing LogDistributions---------------------------")
        if True:
            #Assume missing words are simply very rare
            #So, assign minimum probability to very small elements (e.g. 0 elements)
            distros=np.where(distros>=min_prob,distros,min_prob)
            #Also need to consider minimum probability for "not" distribution
            distros=np.where(distros<=(1-min_prob),distros,1-min_prob)

            return np.log(distros), np.log(1-distros)
        else:
            #Ignore missing words (assume they have P==1, i.e. force log 0 to 0)
            return np.log(np.where(distros>0,distros,1)), np.log(np.where(distros<1,1-distros,1))

    def predict(self,words):
        print("---------------------------Began Classifiying Current Sample---------------------------")
        label = 0 
        negWords = 1 - words
        absPos = negWords* self.logProbWordAbsentGivenPositive
        absNeg = negWords* self.logProbWordAbsentGivenNegative
        pos = np.sum(words* self.logProbWordPresentGivenPositive )+np.sum(absPos) +self.logPriorPositive #do this
        neg = np.sum(words * self.logProbWordPresentGivenNegative  )+np.sum(absNeg)+self.logPriorNegative
        if neg> pos:
            label = -1
        else:
            label =1
        return label

    def test(self): 
        predictions = []
        print("---------------------------Began Testing---------------------------")
        for row in self.xTest: 
            predictions.append(self.predict(row,self.logProbWordPresentGivenPositive, self.logProbWordAbsentGivenPositive, 
                    self.logProbWordPresentGivenNegative, self.logProbWordAbsentGivenNegative, 
                    self.logPriorPositive, self.logPriorNegative))
        
        avgErr = 1-np.average(predictions == self.yTest)
        print("---------------------------Testing Completed---------------------------")
        #returns the loss
        return avgErr

    def skTest(self,xTrain,yTrain,xTest,yTest):
        model = BernoulliNB()
        model.fit(xTrain,yTrain)
        pred = np.array([model.predict(xTest)])
        score = model.score(xTest,yTest)
        return score

class LogisticRegression:
    def __init__(self,learning_rate,num_epoches,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------Logistic Regression---------------------------")
        self.xTrain,self.yTrain, self.xTest,self.yTest = trainFeature,trainLabel,testFeature,testLabel
        self.costs= []
        self.bias = 0
        self.lr = learning_rate
        self.num_epoches = num_epoches
        self.w = None 



    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
    def train(self):
        print("---------------------------Training has started---------------------------")
        feature = self.xTrain
        label = self.yTrain
        self.w = np.ones(feature.shape[1])
        for i in range(self.num_epoches):
            if(i%10==0):
                print(f"Training on Epcoh {i}")
            z = np.dot(feature, self.w) + self.bias
            yHat = self.sigmoid(z)
            error = (yHat - label)
            # compute gradients
            dw = np.average(np.dot(feature.T, error))
            db = np.average(np.sum(error))

            # update parameters
            self.w -= self.lr * dw
            self.bias -= self.lr * db
            self.costs.append(self.cost(feature,label))
        with open("logisticRegression.pickle", "wb") as f:
            model = (self.w,self.bias,self.costs)
            pickle.dump(model, f) 
    def cost(self,feature,label):
        yHat = self.sigmoid(np.dot(feature,self.w)+self.bias)
        cost = - np.average(np.log(label*yHat+(1-label)*(1-yHat)))
        return cost
    def score(self):
        prediciton =self.predict(self.xTest)
        return np.average(prediciton==self.yTest)
    def predict(self, feature):
        prediciton = self.sigmoid(np.dot(feature, self.w)+ self.bias)
        if isinstance(prediciton,np.float64):
            return round(prediciton)
        return  np.array([1 if i > 0.5 else 0 for i in prediciton])



#vocab_size, embedding_dim, rnn_units, batch_size
class RNN:
    def __init__(self):
        pickle_in = open("vocab.pickle","rb")
        self.vocab = pickle.load(pickle_in)
        self.vocab_size = len(self.vocab)
        self.embedding_dim = 256
        self.rnn_units = 1024
        self.batch_size= 64
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        
    def loss(self,y,y_hat):
        computed_loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True) # TODO
        return computed_loss

    def vectorize_comments(self,comments):
        char2idx = {u:i for i ,u in enumerate(self.vocab)}
        return np.array([char2idx[char] for char in comments])

    def get_batch( self,vectorized_comment,seq_length, batch_size):
        # the length of the vectorized songs string
        n = vectorized_comment.shape[0] - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n-seq_length, batch_size)

        '''TODO: construct a list of input sequences for the training batch'''
        input_batch = [vectorized_comment[i:i+seq_length] for i in idx]
        '''TODO: construct a list of output sequences for the training batch'''
        output_batch =  [vectorized_comment[i+1:i+seq_length+1] for i in idx]
        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch
            
    def train(self,feature,isSarcastic):
        if(isSarcastic):
            checkpoint_dir = './sarcastic/training_checkpoints'
        else:
            checkpoint_dir = './notSarcastic/training_checkpoints'
        checkpoint_prefix  = os.path.join(checkpoint_dir, "my_ckpt")
        num_iterations = 5000
        history = []
        plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
        vectorize_comments = self.vectorize_comments(feature)
        for i in range(num_iterations):
            x,y = self.get_batch(vectorize_comments,seq_length= 100,batch_size=self.batch_size)
            loss = self.train_step(x,y)
            history.append(loss.numpy().mean())
            plotter.plot(history)
            if i % 100 == 0:
                print(f"Saving checkpoint {i}") 
                self.model.save_weights(checkpoint_prefix)
    @tf.function
    def train_step(self,x,y):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
        
            '''TODO: feed the current input into the model and generate predictions'''
            y_hat = self.model(x)
        
            '''TODO: compute the loss!'''
            loss = self.loss(y, y_hat)

        # Now, compute the gradients 
        '''TODO: complete the function call for gradient computation. 
            Remember that we want the gradient of the loss with respect all 
            of the model parameters. 
            HINT: use `model.trainable_variables` to get a list of all model
            parameters.'''
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply the gradients to the optimizer so it can update the model accordingly
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def build_model(self):
        model = tf.keras.Sequential([
         tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[self.batch_size, None]),
         self.LSTM(rnn_units=self.rnn_units),

         tf.keras.layers.Dense(self.vocab_size)
         ])
        return model
    
    def LSTM(self,rnn_units):
        return tf.keras.layers.LSTM(
            rnn_units, 
            return_sequences=True, 
            recurrent_initializer='glorot_uniform',
            recurrent_activation='sigmoid',
            stateful=True,
        )
    
    def generate_text(self,isSarcastic, start_string, generation_length=1000):
        char2idx = {u:i for i ,u in enumerate(self.vocab)}
        input_eval = [char2idx[char] for char in start_string]
        print(input_eval)
        input_eval = tf.expand_dims(input_eval,0)
        idx2char = np.array(self.vocab)
        if(isSarcastic):
            checkpoint_dir = './sarcastic/training_checkpoints'
            self.batch_size = 1
            self.model = self.build_model()
            self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
            self.model.build(tf.TensorShape([1, None]))

        else:
            checkpoint_dir = './notSarcastic/training_checkpoints'
            self.batch_size = 1
            self.model = self.build_model()
            self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
            self.model.build(tf.TensorShape([1, None]))
        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        self.model.reset_states()
        tqdm._instances.clear()
        print()
        print("-----------------------------Prediction Started-----------------------------")
        print()
        for i in tqdm(range(generation_length)):
            '''TODO: evaluate the inputs and generate the next character predictions'''
            predictions = self.model(input_eval)
            
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            
            '''TODO: use a multinomial distribution to sample'''
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            # Pass the prediction along with the previous hidden state
            #   as the next inputs to the model
            input_eval = tf.expand_dims([predicted_id], 0)
            
            '''TODO: add the predicted character to the generated text!'''
            # Hint: consider what format the prediction is in vs. the output
            text_generated.append(idx2char[predicted_id])
            
        return (start_string + ''.join(text_generated))

class VotingClassifer:
    def __init__(self,learning_rate,num_epoches,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------Voting Classifer---------------------------")
        lr  = LogisticRegression(learning_rate=learning_rate,num_epoches=num_epochs,trainFeature=trainFeature,trainLabel=trainLabel,testFeature= testFeature, testLabel = testLabel)
        nb = NaiveBaynes(xTrain,yTrain,xTest,yTest)
        pr = Preceptron(learning_rate=learning_rate,num_epoches=num_epochs,trainFeature=trainFeature,trainLabel=trainLabel,testFeature= testFeature, testLabel = testLabel)
    def train(self):
        self.lr.train()
        self.pr.train()
    def predict(self,feature):
        p_lr =self.lr.predict(feature)
        p_nb= self.nb.predict(feature)
        p_pr = self.pr.predict(feature)
        return (p_lr*p_nb)+(p_lr*p_pr)+(p_pr*p_nb)

data = Data()

SARCASM = "WOw thats SOO cooOOl"
NOT_SARACASM  = "This is not sarcasm"

xTrain,yTrain,xTest,yTest = data.get_train_test_data()
learning_rate = 1e-5
num_epochs = 100


vector_for_sarcasm = data.vectorize_input(SARCASM)
vector_for_not_sarcasm= data.vectorize_input(NOT_SARACASM)

classifer = VotingClassifer(learning_rate=learning_rate,num_epoches=num_epochs,trainFeature=xTrain,trainLabel=yTrain,testFeature= xTest, testLabel = yTest)

## if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)



# print(f"Our prediction is {prediciton} ")
# sarcastic_comments = data.get_sarcastic_comments()
# not_sarcastic_comments = data.get_not_sarcastic_comments()
# rnn = RNN()
# rnn.train(not_sarcastic_comments,False)
# rnn.train(sarcastic_comments,True)

# sarcasm = rnn.generate_text(isSarcastic = True,start_string="Hello",generation_length=15)
# print(sarcasm)