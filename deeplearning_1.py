#Install tensor flow in mac https://www.tensorflow.org/install/install_mac 
#pip3 install tensorflow
#If that does not work then
#sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.2.1-py3-none-any.whl
#Run it to check whether it is installed or not


#We're going to be working first with the MNIST dataset, which is a dataset that contains 60,000 training samples 
#and 10,000 testing samples of hand-written and labeled digits, 0 through 9, so ten total "classes." 

#The MNIST dataset has the images, which we'll be working with as purely black and white, thresholded, images, 
#of size 28 x 28, or 784 pixels total. Our features will be the pixel values for each pixel, thresholded. 
#Either the pixel is "blank" (nothing there, a 0), or there is something there (1).

#input --> add weight --> hidden layer 1 (activation function) --> o/p H1
#o/p H1 --> add weight --> hidden layer 1 (activation function) --> o/p H2
#o/p H1 --> add weight --> o/p 

#Compare output to intended output > cost function (cross entropy)
#optimization function(optimizer) > minimize cost (AdamOptimizer ... SGD, AdaGrad)

#backpropagation
#feed forward + backprop = epoc



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#3 hidden layers, and each has 500 nodes.The number 500 needs not to be same.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100#

x = tf.placeholder('float',[None,28*28])
y = tf.placeholder('float') 


def neural_network_model(data):
    
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
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
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x : epoch_x, y : epoch_y})#c is cost
                epoch_loss += c
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
                
        
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

        
train_neural_network(x)