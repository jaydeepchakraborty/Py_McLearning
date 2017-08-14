#This is an example of multi layer neural network on mnist data 

# This is an example of multiple layer neural network on mnist data 
#but the weights are initialized to random - good result

#sudo pip3 install tensorflow
#sudo pip3 install tensorflow-gpu
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#input X: 28*28 greyscale images, the first dimension -100 will index the images in mini-batch
X = tf.placeholder(tf.float32, [100, 784])
# placeholder for correct labels [0--9]
Y_ = tf.placeholder(tf.float32, [None, 10])

pkeep = tf.placeholder(tf.float32)


#weights and biases and initially they are assigned randomly
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))
b3 = tf.Variable(tf.zeros([50]))
W4 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
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

for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y_: batch_Y, pkeep:0.75}

    # train
    sess.run(train_step, feed_dict=train_data)
    
a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

print("Accuracy:- "+str(a))
print("Error Cross Entropy:- "+str(c))