#Tensorflow LSTM for sentiment analysis - example2
# https://github.com/Currie32/Movie-Reviews-Sentiment/blob/master/RNN_movie_reviews.ipynb

# # Predicting the Sentiment of Movie Reviews

# There are two goals for this analysis. The first is to accurately predict the sentiment of movie reviews, 
#and the second is to develop my model in such a way that its outputs can be analyzed with TensorBoard. 
#This is the first time that I am using TensorBoard, so I want to have a somewhat challenging task, and 
#not use a huge dataset. There are 25,000 training and testing reviews, so this model can train multiple 
#iterations overnight on my MacBook Pro. The data is provided by a Kaggle competition from 2015 
#(https://www.kaggle.com/c/word2vec-nlp-tutorial). Despite it having concluded, it can still be used as 
#an excellent learning opportunity. The sections of this analysis are:
# - Inspect the Data
# - Clean and Format the Data
# - Build and Train the Model
# - Make the Predictions
# - Summary

import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from utilsLSTM import clean_text,build_rnn,make_predictions

# Load the data
train = pd.read_csv("data/labeledTrainData.tsv", delimiter="\t")
test = pd.read_csv("data/testData.tsv", delimiter="\t")

#Train data
# print(train.shape)#(25000, 3)
# print(train.head(3))
#     id            sentiment             review
# 0  5814_8          1            With all this stuff going down at the moment w...
# 1  2381_9          1            \The Classic War of the Worlds\" by Timothy Hi...
# 2  7759_3          0            The film starts with a manager (Nicholas Bell)...

#Test data
# print(test.shape)#(25000, 2)
# print(test.head())
#     id                review
# 0  12311_10  Naturally in a film who's main themes are of m...
# 1    8348_2  This movie is a disaster within a disaster fil...
# 2    5828_4  All in all, this is a movie for kids. We saw i...

#Cleaning train data
train_clean = []
for review in train.review:
    train_clean.append(clean_text(review))

#Cleaning test data
test_clean = []
for review in test.review:
    test_clean.append(clean_text(review))
    
#Check how clean data looks like
# for i in range(3):
#     print(train_clean[i])
#     print()
#stuff going moment mj i ve started listening music watching odd documentary
#the classic war worlds timothy hines entertaining film obviously goes great
#film starts manager nicholas bell giving welcome investors robert carradine

# Tokenize the reviews https://keras.io/preprocessing/text/ (specific to keras)
all_reviews = train_clean + test_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)#Fitting is complete.

train_seq = tokenizer.texts_to_sequences(train_clean)

test_seq = tokenizer.texts_to_sequences(test_clean)


# Inspect the reviews after they have been tokenized
#It has total 99426 unique words and it assigns number to them
#then texts_to_sequences gives the number for training dataset
# for i in range(3):
#     print(train_seq[i])
#     print()
#[445, 86, 489, 10867, 8, 61, 582, 2602, 120, 68, 957, 560, 53, 212, 24385, 212, 17035, 219, 193,...
#[10, 279, 218, 3559, 4517, 10472, 355, 3, 460, 188, 22, 708, 8416, 11002, 8651, 1862, 1090, 4607,..
#[3, 464, 2887, 4781, 3622, 669, 2492, 17036, 537, 4483, 10242, 1157, 937, 1062, 31272, 10242,...

## Find the number of unique tokens
# print("Words in index: %d" % len(tokenizer.word_index))#99426


# Find the length of reviews
# lengths = []
# for review in train_seq:
#     lengths.append(len(review))
# 
# for review in test_seq:
#     lengths.append(len(review))

# Create a dataframe so that the values can be inspected
# lengths = pd.DataFrame(lengths, columns=['counts'])
# print(lengths.counts.describe())
# count    50000.000000
# mean       131.048860
# std         98.409141
# min          3.000000
# 25%         70.000000
# 50%         98.000000
# 75%        160.000000
# max       1476.000000
# Name: counts, dtype: float64


# print(np.percentile(lengths.counts, 80))#182.0 --> it means 80% of the reviews has max length 182
# print(np.percentile(lengths.counts, 85))#213.0
# print(np.percentile(lengths.counts, 90))#258.0
# print(np.percentile(lengths.counts, 95))#338.0
#the above stat will help me to find to chose the review length, I choose 260, so I covered 90% of the reviews

# Pad and truncate the questions so that they all have the same length.
max_review_length = 260

train_pad = pad_sequences(train_seq, maxlen = max_review_length) #specific to keras
test_pad = pad_sequences(test_seq, maxlen = max_review_length)


# for i in range(1):
#     print(len(train_seq[i]))
#     print(train_pad[i,:])
#     print()
#first train string length is 237, and pad_sequences padded first (260-237)
#23 with zeros
# [    0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0   445
#     86   489 10867     8    61   582  2602   120    68   957   560    53
#    212 24385   212 17035   219   193    97    20   695  2563   124   109
#     15   520  3942   193    27   246   654  2349  1261 17035    90  4780
#     90   712     3   305    86    16   358  1839   542  1219  3576 10867 .. ]


# Creating the training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size = 0.15, random_state = 2)
x_test = test_pad

# Inspect the shape of the data
# print(x_train.shape)#(21250, 260)
# print(x_valid.shape)#(3750, 260)
# print(x_test.shape)#(25000, 260)


# The default parameters of the model
n_words = len(tokenizer.word_index)
embed_size = 300
batch_size = 250
lstm_size = 128
num_layers = 2
dropout = 0.5
learning_rate = 0.001
epochs = 100
multiple_fc = False
fc_units = 256


# Train the model with the desired tuning parameters
# There will be 10 check points/ models. The models will be saved in the following path
#checkpoint_folder = "/Users/jaydeep/jaydeep_workstation/Workplace/Python/PyMcLearning/data/chkpoint/lstm/sentiment/" 
for lstm_size in [64,128]:
    for multiple_fc in [True, False]:
        for fc_units in [128, 256]:
            log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,multiple_fc,fc_units)
            model = build_rnn(n_words = n_words, 
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = lstm_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = fc_units)            
            train(model, epochs, x_train, y_train, x_valid, y_valid, batch_size, dropout, log_string)

checkpoint_folder = "/Users/jaydeep/jaydeep_workstation/Workplace/Python/PyMcLearning/data/chkpoint/lstm/sentiment/" 
#you can choose any one saved checkpoint or more than one
#For this example, we are using two checkpoints
lstm_size = 64 
multiple_fc = False
fc_units = 256
log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,multiple_fc,fc_units)
checkpoint1 = checkpoint_folder+"sentiment_{}.ckpt".format(log_string)
model = build_rnn(n_words = n_words, 
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = lstm_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = fc_units) 
predictions1 = make_predictions(model,x_test, batch_size, checkpoint1)

lstm_size = 64 
multiple_fc = True
fc_units = 256
log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,multiple_fc,fc_units)
checkpoint2 = checkpoint_folder+"sentiment_{}.ckpt".format(log_string)
model = build_rnn(n_words = n_words, 
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = lstm_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = fc_units) 
predictions2 = make_predictions(model,x_test, batch_size, checkpoint2)


# Average the best two predictions
predictions_combined = (pd.DataFrame(predictions1) + pd.DataFrame(predictions2))/2

#saving the output
submission = pd.DataFrame(data={"id":test["id"], "sentiment":predictions_combined})
submission.to_csv("submission_{}.csv".format("output"), index=False, quoting=3)

