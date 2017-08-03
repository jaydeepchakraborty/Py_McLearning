#This is an example of single layer neural network on iris data

#sudo pip3 install tensorflow
#sudo pip3 install tensorflow-gpu
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 42

def get_iris_data():
    #3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length and width
    iris = datasets.load_iris()
    data = iris['data'] #vectors or features petal_length,petal_width,sepal_length,sepal_width
    target = iris['target'] #labels
    
    
    N,M = data.shape #N rows =150, M columns =4
    
    #It creates an array with 150*5 and filled with value 1
    all_X = np.ones((N, M + 1))
#     [[ 1.   1.  1.  1.  1.]
#      [ 1.   1.  1.  1.  1.]
#      [ 1.   1.  1.  1.  1.]
#      [ 1.   1.  1.  1.  1.]...]
    
    
    
    #It copied the data into all_X but the first column will be intact
    #i.e no change in the first column
    all_X[:, 1:] = data
#     [[ 1.   5.1  3.5  1.4  0.2]
#      [ 1.   4.9  3.   1.4  0.2]
#      [ 1.   4.7  3.2  1.3  0.2]
#      [ 1.   4.6  3.1  1.5  0.2]...]


    num_lbls = len(np.unique(target))#you will get 3 --> Setosa = 0, Versicolour = 1, and Virginica = 2
    
#     In our neural network 
#     0 => [ 1.  0.  0.]
#     1 => [ 0.  1.  0.]
#     1 => [ 0.  0.  1.]
    all_Y = np.eye(num_lbls)[target]

    return train_test_split(all_X,all_Y, test_size=0.33, random_state=RANDOM_SEED)



def main():
    train_X, test_X, train_Y, test_Y = get_iris_data()
    
    x_size = train_X.shape[1]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()