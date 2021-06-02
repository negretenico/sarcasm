from numpy.lib.function_base import vectorize
import pandas as pd 
import numpy as np 
import pickle
from sklearn.naive_bayes import BernoulliNB
np.seterr(divide='ignore', invalid='ignore')
class Preceptron:
    def __init__(self,learning_rate,num_epoches,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------Preceptron---------------------------")
        self.learning_rate = learning_rate
        self.num_epoches = num_epoches
        self.xTrain,self.yTrain, self.xTest,self.yTest = trainFeature,trainLabel,testFeature,testLabel
        self.weights = np.zeros(np.shape(self.xTrain)[1]+1)

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
            model = self
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
            model = self
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
            model = self
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
class VotingClassifer:
    def __init__(self,learning_rate,num_epoches,trainFeature,trainLabel,testFeature,testLabel):
        print("---------------------------Voting Classifer---------------------------")
        self.lr  = LogisticRegression(learning_rate=learning_rate,num_epoches=num_epoches,trainFeature=trainFeature,trainLabel=trainLabel,testFeature= testFeature, testLabel = testLabel)
        self.nb = NaiveBaynes(trainFeature,trainLabel,testFeature,testLabel)
        self.pr = Preceptron(learning_rate=learning_rate,num_epoches=num_epoches,trainFeature=trainFeature,trainLabel=trainLabel,testFeature= testFeature, testLabel = testLabel)
    def train(self):
        self.lr.train()
        self.pr.train()
    def predict(self,feature):
        p_lr =self.lr.predict(feature)
        p_nb= self.nb.predict(feature)
        p_pr = self.pr.predict([feature])[0]
        return (p_lr*p_nb)+(p_lr*p_pr)+(p_pr*p_nb)
