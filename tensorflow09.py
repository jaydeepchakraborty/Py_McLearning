#sentiment analysis using deep neural network
#check deeplearning_2

#https://www.youtube.com/watch?v=RZYjsw6P4nI
#https://www.youtube.com/watch?v=bPYJi1E9xeM
#https://www.youtube.com/watch?v=7fcWfUavO7E&index=5&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP
#imports
from collections import Counter
import pickle
import random
import traceback

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words

import numpy as np
import tensorflow as tf


def loadFile():
    data =[]
    target = []
    with open("data/pos.txt",'r', encoding="utf-8") as f:
        pos_contents = f.readlines()
        pos_contents = [x.strip() for x in pos_contents]
        t_val = [0,1]
        pos_target = [t_val] * len(pos_contents)
        
    with open("data/neg.txt",'r', encoding="utf-8") as f:
        neg_contents = f.readlines()
        neg_contents = [x.strip() for x in neg_contents]
        t_val = [1,0]
        neg_target = [t_val] * len(neg_contents)


    data = pos_contents + neg_contents
    target = pos_target + neg_target
    
    return data,target

def preprocess(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_data = tfidf_vectorizer.fit_transform(data)
    #tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data.toarray()

try:
    data,target = loadFile()
    tf_idf = preprocess(data)
     
    train_x,test_x,train_y,test_y = train_test_split(tf_idf,target,test_size=0.4,random_state=43,shuffle=True)
    
    n_classes = 2
    batch_size = 100#
    
    x = tf.placeholder('float',[None,len(train_x[0])])
    y_ = tf.placeholder('float')
    pkeep = tf.placeholder(tf.float32)
    
    W1 = tf.Variable(tf.truncated_normal([len(train_x[0]), 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]))
    W2 = tf.Variable(tf.truncated_normal([500, 500], stddev=0.1))
    b2 = tf.Variable(tf.zeros([500]))
    W3 = tf.Variable(tf.truncated_normal([500, n_classes], stddev=0.1))
    b3 = tf.Variable(tf.zeros([n_classes]))
    
    # model 3-layer neural network.
    xx = x
    Y1 = tf.nn.sigmoid(tf.matmul(xx, W1) + b1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + b2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Ylogits = tf.matmul(Y2d, W3) + b3
    y = tf.nn.softmax(Ylogits)#softmax function for multiclass classification
    
    #loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    
    #optimiser -
    optimizer = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
    
    #calculating accuracy of our model 
    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #OR we can use 
    is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    #session parameters
    sess = tf.InteractiveSession()
    #initialising variables
    init = tf.global_variables_initializer()
    sess.run(init)
    hm_epochs = 100
    
    print("9")
    for epoch in range(hm_epochs):
        epoch_loss = 0;
        
        i = 0
        while i < len(train_x):
            start = i
            end = i+ batch_size
            
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer,cross_entropy], feed_dict = {x : batch_x, y_ : batch_y, pkeep:1})#c is cost
            epoch_loss += c
            i +=  batch_size
        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
    
    #finding accuracy on testing set
    feed_dict_test = {x: np.array(test_x),
                      y_: np.array(test_y), pkeep:1} 
    
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    
    print("Accuracy:- "+str(acc))
    
    
except:
    traceback.print_exc()
    
    
    
    
    