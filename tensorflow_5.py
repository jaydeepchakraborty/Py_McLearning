# This is an example of deep neural network on custom data

# This is an example of convolution neural network on custom data
#check also cnn_2 
#This is Kaggle's cat-dog problem
#source https://github.com/Vikramank/Deep-Learning-/blob/master/Cats-and-Dogs/Classification-%20Cats%20and%20Dogs.ipynb
#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
#for writing text files
import glob
import os     
import random 
#reading images from a text file
from tflearn.data_utils import image_preloader
import math


##################Data processing###########################
IMAGE_FOLDER = 'data/train'
TRAIN_DATA = 'data/training_data.txt'
TEST_DATA = 'data/test_data.txt'
VALIDATION_DATA = 'data/validation_data.txt'
train_proportion=0.7
test_proportion=0.2
validation_proportion=0.1

#read the image directories
filenames_image = os.listdir(IMAGE_FOLDER)

#shuffling the data is important otherwise the model will be fed with a single class data for a long time and 
#network will not learn properly
random.shuffle(filenames_image)

#total number of images
total=len(filenames_image)

##  *****training data******** 
fr = open(TRAIN_DATA, 'w')
train_files=filenames_image[0: int(train_proportion*total)]
for filename in train_files:
    if filename[0:3] == 'cat':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')

fr.close()

##  *****testing data******** 
fr = open(TEST_DATA, 'w')
test_files=filenames_image[int(math.ceil(train_proportion*total)):int(math.ceil((train_proportion+test_proportion)*total))]
for filename in test_files:
    if filename[0:3] == 'cat':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
fr.close()


##  *****validation data******** 
fr = open(VALIDATION_DATA, 'w')
valid_files=filenames_image[int(math.ceil((train_proportion+test_proportion)*total)):total]
for filename in valid_files:
    if filename[0:3] == 'cat':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:3] == 'dog':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
fr.close()


#importing data
X_train, Y_train = image_preloader(TRAIN_DATA, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
X_test, Y_test = image_preloader(TEST_DATA, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)
X_val, Y_val = image_preloader(VALIDATION_DATA, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)


print("Dataset")
print("Number of training images {}".format(len(X_train)))
print("Number of testing images {}".format(len(X_test)))
print("Number of validation images {}".format(len(X_val)))
print("Shape of an image {}" .format(X_train[1].shape))
print("Shape of label:{} ,number of classes: {}".format(Y_train[1].shape,len(Y_train[1])))



X=tf.placeholder(tf.float32,shape=[None,56,56,3] , name='input_image') 
Y_=tf.placeholder(tf.float32,shape=[None, 2] , name='input_class')
lr = tf.placeholder(tf.float32)# variable learning rate
pkeep = tf.placeholder(tf.float32)

#weights and biases and initially they are assigned randomly
W1 = tf.Variable(tf.truncated_normal([56*56*3, 5000], stddev=0.1))
b1 = tf.Variable(tf.zeros([5000]))
W2 = tf.Variable(tf.truncated_normal([5000, 1000], stddev=0.1))
b2 = tf.Variable(tf.zeros([1000]))
W3 = tf.Variable(tf.truncated_normal([1000, 50], stddev=0.1))
b3 = tf.Variable(tf.zeros([50]))
W4 = tf.Variable(tf.truncated_normal([50, 2], stddev=0.1))
b4 = tf.Variable(tf.zeros([2]))


# model 4-layer neural network.
xx = tf.reshape(X, [-1, 56*56*3])
Y1 = tf.nn.sigmoid(tf.matmul(xx, W1) + b1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + b2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.sigmoid(tf.matmul(Y2d, W3) + b3)
Y3d = tf.nn.dropout(Y3, pkeep)
Ylogits = tf.matmul(Y3d, W4) + b4
Y = tf.nn.softmax(Ylogits)



# loss function provided by Tensorflow
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#we can use different type of optimizer
#before it we have used GradientDescentOptimizer
#Now we are using AdamOptimizer
optimizer = tf.train.AdamOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


learning_rate = 0.003

epoch = 5000
batch_size = 20
previous_batch = 0

for i in range(epoch):
    #batch wise training 
    current_batch=previous_batch+batch_size
    if current_batch >= len(X_train) : 
        x_input=np.asarray(X_train[previous_batch:len(X_train)])
#         current_batch=batch_size-(len(X_train)-previous_batch)
        x_images=np.reshape(x_input,[batch_size,56,56,3])
        
        y_input=np.asarray(Y_train[previous_batch:len(X_train)])
        y_label=np.reshape(y_input,[batch_size,2])
        
        current_batch= 0
    else :
        x_input=X_train[previous_batch:current_batch]
        x_images=np.reshape(x_input,[batch_size,56,56,3])
        y_input=Y_train[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,2])
    previous_batch=current_batch

    _,loss=sess.run([train_step, cross_entropy], feed_dict={X: x_images,Y_: y_label, lr: learning_rate, pkeep:0.95})


#here we have done for only one TEST data, in real case we have to do Cross-Validation

x_test_images=np.reshape(np.asarray(X_test),[len(X_test),56,56,3])
y_test_labels=np.reshape(np.asarray(Y_test),[len(Y_test),2]) 
test_accuracy,test_loss=sess.run([accuracy, cross_entropy],feed_dict={X: x_test_images ,Y_: y_test_labels, lr: learning_rate, pkeep:0.95})
print("TEST_DATA, Accuracy:{} , Loss : {}" .format(test_accuracy,test_loss))

 


v_test_images=np.reshape(np.asarray(X_val),[len(X_val),56,56,3])
v_test_labels=np.reshape(np.asarray(Y_val),[len(Y_val),2]) 
valid_accuracy,valid_loss=sess.run([accuracy, cross_entropy],feed_dict={X: v_test_images ,Y_: v_test_labels, lr: learning_rate, pkeep:0.95})
print("VALIDATION_DATA, Accuracy:{} , Loss : {}" .format(valid_accuracy,valid_loss))


