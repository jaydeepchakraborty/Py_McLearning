#Install tensor flow in mac https://www.tensorflow.org/install/install_mac 
#pip3 install tensorflow
#If that does not work then
#sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.2.1-py3-none-any.whl
#Run it to check whether it is installed or not

#Sentiment analysis : here data is not tensor data
# we have downloaded the data from the following link
#https://pythonprogramming.net/using-our-own-data-tensorflow-deep-learning-tutorial/?completed=/tensorflow-neural-network-session-machine-learning-tutorial/
# It has two txt files 1) pos, 2) neg



import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
hm_lines = 10000000

def create_lexicon(pos,neg):
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
    
    
    return filtered_lexicon

def sample_handling(sample,lexicon,classification):

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

    return featureset


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
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

    return train_x,train_y,test_x,test_y


def saveModel():
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    # if you want to pickle this data:
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
        print('saved')
 

#saveModel()



train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

#3 hidden layers, and each has 500 nodes.The number 500 needs not to be same.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100#

x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float') 


def neural_network_model(data):
    
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    ouptput_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    #(input_data * weights) + biases
    
    l1 = tf.add(tf.matmul(data,hidden_layer1['weights']) , hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)#activation function
    
    l2 = tf.add(tf.matmul(l1,hidden_layer2['weights']) , hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)#activation function
    
    l3 = tf.add(tf.matmul(l2,hidden_layer3['weights']) , hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)#activation function
    
    output = tf.add(tf.matmul(l3,ouptput_layer['weights']) , ouptput_layer['biases'])
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)#learning_rate = 0.001
    hm_epochs = 5
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0;
            
            i = 0
            while i < len(train_x):
                start = i
                end = i+ batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer,cost], feed_dict = {x : batch_x, y : batch_y})#c is cost
                epoch_loss += c
                i +=  batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

        
train_neural_network(x)



