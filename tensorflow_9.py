#sentiment analysis using deep neural network
#check deeplearning_2

#imports
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
from nltk.corpus import stopwords
from stop_words import get_stop_words
import traceback

#constants
pos = "data/pos.txt"
neg = "data/neg.txt"

lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
hm_lines = 100000


def create_lexicon(pos,neg):
    print("3")
    lexicon = []
    for f in [pos,neg]:
        with open(f,'r', encoding="utf-8") as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
                
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    filtered_lexicon = []
    
    for w in lexicon:
        if w not in stopWords:
            filtered_lexicon.append(w) 
    
    print("4")
    return filtered_lexicon


def sample_handling(sample,lexicon,classification):
    print("5")

    featureset = []

    with open(sample,'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features,classification])
    print("6")
    return featureset

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    print("2")
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0]) #for positive class is [1,0]
    features += sample_handling('neg.txt',lexicon,[0,1])#for negative class is [0,1]
    random.shuffle(features)
    features = np.array(features)
    
    

    testing_size = int(test_size*len(features))#10% of features

    train_x = list(features[:,0][:-testing_size])# all the features up to testing_size 
    train_y = list(features[:,1][:-testing_size])# all the labels up to testing_size 
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    print("7")
    return train_x,train_y,test_x,test_y

try:
    print("1")
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels(pos,neg)
    print("8")
    
    #2 hidden layers, and each has 500 nodes.The number 500 needs not to be same.
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    
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
    hm_epochs = 1000
    
    print("9")
    for epoch in range(hm_epochs):
        epoch_loss = 0;
        
        i = 0
        while i < len(train_x):
            start = i
            end = i+ batch_size
            
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer,cross_entropy], feed_dict = {x : batch_x, y : batch_y, pkeep:1})#c is cost
            epoch_loss += c
            i +=  batch_size
        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
    
    #finding accuracy on testing set
    feed_dict_test = {x: np.array(test_x),
                      y_: np.array(test_y), pkeep:1} 
    
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    
    print("Accuracy:- "+str(acc))
    print("10")
    
except:
    traceback.print_exc()
    
    
    
    
    