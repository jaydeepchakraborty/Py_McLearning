#source : https://github.com/Vikramank/Deep-Learning-/blob/master/Iris%20data%20classification.ipynb

import tensorflow as tf
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt


#import data 
data=pd.read_csv('data/iris.csv', names=['f1','f2','f3','f4','f5'], skiprows=1)
#We have three labels in f5
#1) Iris-setosa, 2) Iris-versicolor 3) Iris-virginica
label_count = data["f5"].value_counts()
# print(label_count)
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50
# Name: f5, dtype: int64

#map data into arrays, #specifically for deep learning
s=np.asarray([1,0,0])
ve=np.asarray([0,1,0])
vi=np.asarray([0,0,1])
data['f5'] = data['f5'].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})
#Now we have three labels in f5
#1) [1,0,0], 2) [0,1,0] 3) [0,0,1]

#shuffle the data
data=data.iloc[np.random.permutation(len(data))]

#         f1     f2     f3     f4       f5
# 22     4.6    3.6    1.0    0.2    [1, 0, 0]
# 138    6.0    3.0    4.8    1.8    [0, 0, 1]
# 94     5.6    2.7    4.2    1.3    [0, 1, 0]
# ..........

#THe above line shuffles the data but it will intact the ID for each data row.
#Now the following line will reset the index and index will be 0,1,2..150 
#but the data will still be shuffled
data=data.reset_index(drop=True)
#         f1     f2     f3     f4       f5
# 0     4.6    3.6    1.0    0.2    [1, 0, 0]
# 1    6.0    3.0    4.8    1.8    [0, 0, 1]
# 2     5.6    2.7    4.2    1.3    [0, 1, 0]
# ..........

label=data['f5']
#training data
x_input=data.ix[0:105,['f1','f2','f3','f4']]
y_input=label[0:106]#it does not include the header
#test data
x_test=data.ix[106:149,['f1','f2','f3','f4']]
y_test=label[106:150]#it does not include the header


#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])

pkeep = tf.placeholder(tf.float32)

#Followed https://www.neuraldesigner.com/learning/examples/iris_flowers_classification
#weights and biases and initially they are assigned randomly
W1 = tf.Variable(tf.truncated_normal([4, 5], stddev=0.1))
b1 = tf.Variable(tf.zeros([5]))
W2 = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1))
b2 = tf.Variable(tf.zeros([3]))
W3 = tf.Variable(tf.truncated_normal([3, 3], stddev=0.1))
b3 = tf.Variable(tf.zeros([3]))

# model 3-layer neural network.
xx = x
Y1 = tf.nn.sigmoid(tf.matmul(xx, W1) + b1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + b2)
Y2d = tf.nn.dropout(Y2, pkeep)
Ylogits = tf.matmul(Y2d, W3) + b3
y = tf.nn.softmax(Ylogits)#softmax function for multiclass classification


#loss function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) 
#OR we can use 
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

epoch = 2000
for step in range(epoch):
    feed_dict_train_data = {x: x_input, y_:[t for t in y_input.as_matrix()], pkeep:1}# we kept all the synopsys as we have very less data 
    # train
    sess.run(optimizer, feed_dict=feed_dict_train_data)
    
    
#finding accuracy on testing set
feed_dict_test = {x: x_test,
                  y_: [t for t in y_test.as_matrix()], pkeep:1} 

acc = sess.run(accuracy, feed_dict=feed_dict_test)

print("Accuracy:- "+str(acc))