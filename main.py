from random import paretovariate
from model import Seq2Seq
from dataGenerator import Data


data = Data()

MAX_LENGTH = 64
comment,parent_comment,pairs = data.prepareData('comments', 'responses', True)
model = Seq2Seq(MAX_LENGTH=MAX_LENGTH,comment=comment,parent_comment=parent_comment)
model.train(comment,parent_comment,pairs,n_iters = 75000,print_every=5000)

# train_dataset, val_dataset, comments, responses = data.call( BUFFER_SIZE, BATCH_SIZE)
SARCASM = "WOw thats SOO cooOOl"
print(SARCASM)
reply, _ = model.evaluate(comment,parent_comment,SARCASM,MAX_LENGTH)
print(" ".join(reply))
# NOT_SARACASM  = "This is not sarcasm"

# embedding_dim = 256
# units = 1024
# steps_per_epoch = data.LIMIT//BATCH_SIZE


# xTrain,yTrain,xTest,yTest = data.get_train_test_data()
# learning_rate = 1e-5
# num_epochs = 100


# vector_for_sarcasm = data.vectorize_input(SARCASM)
# vector_for_not_sarcasm= data.vectorize_input(NOT_SARACASM)



# sarcastic_comments = data.get_sarcastic_comments()
# not_sarcastic_comments = data.get_not_sarcastic_comments()
# rnn = RNN()
# rnn.train(not_sarcastic_comments,False)
# rnn.train(sarcastic_comments,True)

# sarcasm = rnn.generate_text(isSarcastic = True,start_string="",generation_length=15)
# print(sarcasm)