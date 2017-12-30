#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb

import tensorflow as tf
import numpy as np
import pandas as pd


###################Data Preparation Start###########################
#import data 
data=pd.read_csv('data/iris.csv', names=['f1','f2','f3','f4','f5'], skiprows=1)

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

###################Data Preparation End###########################

###################Training Parameters Start###########################
learning_rate = 0.0003

# Network Parameters
num_input = 4 # IRIS data input (4 features)
height = 1
width = 4 #num_input same
channel = 1
num_classes = 3 # 3 classes (setosa, versicolor, virginica)
dropout = 1 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
###################Training Parameters End###########################

###################Convolution Model Start###########################

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # In our case it is 4 features (4*1)
    # Reshape to match picture format [Height x Width x Channel]
    # In our case Height = 1, Width = 4, Channel = 1 (for color RGB=3, grey = 1, in our case 1)
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, height, width, channel])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

###################Convolution Model End###########################


###################Weights and Biases Start###########################
# Store layers weight & bias
#28x28x1 -->conv2d-->28x28x32-->maxpool-->14x14x32 -->conv2d-->14x14x64-->maxpool--> 7x7x64
#In our case
#1x4x1 -->conv2d-->1x4x10-->maxpool(no down sampling as k=1)-->1x4x10 -->conv2d-->1x4x20-->maxpool--> 1x4x20
#80 --> FC -->60 --> softmax --> 3
conv1_in = 1
conv1_out = 10 #this u can play with
conv2_out = 20 #this u can play with
fc_out = 60 #this u can play with
conv1_flt_ht = 1
conv1_flt_wd = 3 #this u can play with but never more that number of features
conv2_flt_ht = 1
conv1_flt_wd = 2 #this u can play with but never more that number of features
weights = {
    # 5x5 conv, 1 input, 32 outputs
    #IN our case # 1x3 conv, 1 input, 10 outputs
    'wc1': tf.Variable(tf.random_normal([conv1_flt_ht, 3, conv1_in, conv1_out])),
    # 5x5 conv, 32 inputs, 64 outputs
    #IN our case # 1x2 conv, 10 inputs, 20 outputs
    'wc2': tf.Variable(tf.random_normal([conv1_flt_ht, 2, conv1_out, conv2_out])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    #IN our case # 1x4 conv, 64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([height*width*conv2_out, fc_out])),
    # 1024 inputs, 10 outputs (class prediction)
    # In Our case 60 inputs, 3 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([fc_out, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([conv1_out])),
    'bc2': tf.Variable(tf.random_normal([conv2_out])),
    'bd1': tf.Variable(tf.random_normal([fc_out])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

###################Weights and Biases End###########################



# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    epoch = 2000 # this u can play with
    for step in range(epoch):
        feed_dict_train_data = {X: x_input, Y:[t for t in y_input.as_matrix()], keep_prob:1}# we kept all the synopsys as we have very less data 
        # train
        loss, acc = sess.run([loss_op,accuracy], feed_dict=feed_dict_train_data)
        if(step%100 == 0):
            print("Train Accuracy:- "+str(acc) + " Train Loss:- "+str(loss))
    
    
    #finding accuracy on testing set
    feed_dict_test = {X: x_test,
                      Y: [t for t in y_test.as_matrix()], keep_prob:1} 
    
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    
    print("Test Accuracy:- "+str(acc))
