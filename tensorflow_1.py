#This is an example of single layer neural network on mnist data 

#sudo pip3 install tensorflow
#sudo pip3 install tensorflow-gpu
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))


# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])# true labels
#input layer
X = tf.placeholder(tf.float32, [None, 784]) #28*28 = 784
#weights and biases
W = tf.Variable(tf.zeros([784, 10])) #10 number of classes or output
b = tf.Variable(tf.zeros([10]))


# model 1-layer neural network.
Ylogits = tf.matmul(X, W) + b
Y = tf.nn.softmax(Ylogits)


# loss function
#TensorFlow has a built-in function for calculating the cross-entropy. Note that it uses the values of 
#the logits because it also calculates the softmax internally.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_)
#we simply take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.003)
optimizer = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000): #1000 is epoch
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    feed_dict_train_data={X: batch_X, Y_: batch_Y}

    # train
    sess.run(optimizer, feed_dict=feed_dict_train_data)
 
#finding accuracy on testing set
feed_dict_test = {X: mnist.test.images,
                  Y_: mnist.test.labels} 
   
acc = sess.run(accuracy, feed_dict=feed_dict_test)

print("Accuracy:- "+str(acc))