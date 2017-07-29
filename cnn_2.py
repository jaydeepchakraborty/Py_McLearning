#https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
#CNN on a kaggle competition
# described in https://pythonprogramming.net

import cv2                 # working with, mainly resizing, images ##pip3 install opencv-python
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. ##pip3 install tqdm
import pickle
import matplotlib.pyplot as plt


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = '/Users/jaydeep/jaydeep_workstation/Workplace/Python/PyMcLearning/data/cat_dog/train'
TEST_DIR = '/Users/jaydeep/jaydeep_workstation/Workplace/Python/PyMcLearning/data/cat_dog/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = './dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]


#This function is for creating the data for training
#first we are getting the label of the image. the files are named like this cat.0.jpg, dog.9694.jpg
#so from the file name we will get the label, cat --> [1,0] and  dog --> [0,1]
#Then each image is resized to 50-50 dimension
#then save the data for next purpose
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#This function is for creating the data for testing
#This is the actual competition test data, 
#NOT the data that we'll use to check the accuracy of our algorithm as we test. This data has no label.

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



def crt_neural_network():
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.95)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model
 
 
def fit_data(model):
    
    train_data = np.load('train_data.npy')
    train = train_data[:-500]
    test = train_data[-500:]
     
    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]
     
    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    
    
    model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


def visualize_test_data(model):
    
    test_data = np.load('test_data.npy')
    fig=plt.figure()
    
    #plotting first 12 images
    for num,data in enumerate(test_data[:12]):
        
        y = fig.add_subplot(3,4,num+1)#num must be 1 <= num <= 12
        
        img_num = data[1]
        img_data = data[0]
        orig_data = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        
        if np.argmax(model_out) == 1: str_label='Dog'
        else: str_label='Cat'
        
        y.imshow(orig_data,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()          
        
def mainMtdh():
    #create_train_data()
    #process_test_data()
    model = crt_neural_network()
    
    fit_data(model)
    visualize_test_data(model)
      
 
 
 
mainMtdh()   