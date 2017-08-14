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

# fully connected layer (tha last layer has 2 softmax neurons)DOG/CAT
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer


W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/2)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/2)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/2)
W4 = tf.Variable(tf.truncated_normal([14 * 14 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/2)
W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
B5 = tf.Variable(tf.ones([2])/2)



#convolutional layer 1 --convolution+RELU activation
# The model
stride = 1  # output is 56x56
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 28x28
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 14x14
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 14 * 14 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
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

    _,loss=sess.run([train_step, cross_entropy], feed_dict={X: x_images,Y_: y_label, lr: learning_rate})


#here we have done for only one TEST data, in real case we have to do Cross-Validation

x_test_images=np.reshape(np.asarray(X_test),[len(X_test),56,56,3])
y_test_labels=np.reshape(np.asarray(Y_test),[len(Y_test),2]) 
test_accuracy,test_loss=sess.run([accuracy, cross_entropy],feed_dict={X: x_test_images ,Y_: y_test_labels, lr: learning_rate})
print("TEST_DATA, Accuracy:{} , Loss : {}" .format(test_accuracy,test_loss))

 


v_test_images=np.reshape(np.asarray(X_val),[len(X_val),56,56,3])
v_test_labels=np.reshape(np.asarray(Y_val),[len(Y_val),2]) 
valid_accuracy,valid_loss=sess.run([accuracy, cross_entropy],feed_dict={X: v_test_images ,Y_: v_test_labels, lr: learning_rate})
print("VALIDATION_DATA, Accuracy:{} , Loss : {}" .format(valid_accuracy,valid_loss))


