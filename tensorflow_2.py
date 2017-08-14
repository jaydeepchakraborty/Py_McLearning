# https://www.youtube.com/watch?v=wuo4JdG3SvU&index=1&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
# This is an example of multiple layer neural network on mnist data 
#but the weights are initialized to zeros - bad result

# sudo pip3 install tensorflow
# sudo pip3 install tensorflow-gpu
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

pkeep = tf.placeholder(tf.float32)  # For dropout - used for overfitting

# placeholder for correct labels [0--9]
Y_ = tf.placeholder(tf.float32, [None, 10])
# input X: 28*28 greyscale images, the first dimension -100 will index the images in mini-batch
X = tf.placeholder(tf.float32, [None, 784])


# weights and biases and initially they are assigned zero
W1 = tf.Variable(tf.zeros([784, 200]))
b1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.zeros([200, 100]))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.zeros([100, 50]))
b3 = tf.Variable(tf.zeros([50]))
W4 = tf.Variable(tf.zeros([50, 10]))
b4 = tf.Variable(tf.zeros([10]))


# model 4-layer neural network.
xx = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(xx, W1) + b1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + b2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.sigmoid(tf.matmul(Y2d, W3) + b3)
Y3d = tf.nn.dropout(Y3, pkeep)
Ylogits = tf.matmul(Y3d, W4) + b4
Y = tf.nn.softmax(Ylogits)


# loss function provided by Tensorflow
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cost = tf.reduce_mean(cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# we can use different type of optimizer
# before it we have used GradientDescentOptimizer
# Now we are using AdagradOptimizer, AdamOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.003)
optimizer = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    feed_dict_train_data = {X: batch_X, Y_: batch_Y, pkeep:0.95}

    # train
    sess.run(optimizer, feed_dict=feed_dict_train_data)
    
# finding accuracy on testing set
feed_dict_test = {X: mnist.test.images,
                  Y_: mnist.test.labels,
                pkeep:1.0} #not loosing any neuron

  
acc = sess.run(accuracy, feed_dict=feed_dict_test)

print("Accuracy:- " + str(acc))
