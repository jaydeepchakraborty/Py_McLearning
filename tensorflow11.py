#Train a LSTM to predict the next word using a sample short story
#https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
#https://github.com/seyedsaeidmasoumzadeh/Predict-next-word

import collections
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


# Word embedding
def word_embedding(words):
    vocabulary = collections.Counter(words).most_common()
#Counter('abcdeabcdabcaba').most_common(3) --> [('a', 5), ('b', 4), ('c', 3)]
#     print(vocabulary)
#[(',', 14), ('the', 11), ('.', 8), ('and', 7), ('to', 6), ('said', 6), ... , ('small', 1), ('make', 1)]
    vocabulary_dictionary = dict()
    for word, _ in vocabulary:
        # Assign a numerical unique value to each word inside vocabulary 
        vocabulary_dictionary[word] = len(vocabulary_dictionary)
    rev_vocabulary_dictionary = dict(zip(vocabulary_dictionary.values(), vocabulary_dictionary.keys()))
    
#     print(vocabulary_dictionary)
#{',': 0, 'the': 1, '.': 2, 'and': 3, 'said': 4, 'to': 5, ... , 'small': 110, 'make': 111}
#     print(rev_vocabulary_dictionary)
#{0: ',', 1: 'the', 2: '.', 3: 'and', 4: 'said', 5: 'to', .... , 110: 'small', 111: 'make'}
    return vocabulary_dictionary, rev_vocabulary_dictionary


# Build Training data. For example if X = ['long', 'ago', ','] then Y = ['the']
def sampling(words, vocabulary_dictionary, window):
    X = []
    Y = []
    sample = []
    for index in range(0, len(words) - window):
        for i in range(0, window):
            sample.append(vocabulary_dictionary[words[index + i]])
            if (i + 1) % window == 0:
                X.append(sample)
                Y.append(vocabulary_dictionary[words[index + i + 1]])
                sample = []
    return X,Y


with open("data/tf11_data.txt") as f:
    content = f.read()
#print(content)
#long ago , the mice had a general ... easy to propose impossible remedies .
words = nltk.tokenize.word_tokenize(content)
#print(words)
#['long', 'ago', ',', 'the', 'mice', ... , 'propose', 'impossible', 'remedies', '.']
vocabulary_dictionary, reverse_vocabulary_dictionary = word_embedding(words)

window = 3
num_classes = len(vocabulary_dictionary)
timesteps = window
num_hidden = 512
num_input = 1
batch_size = 20
iteration = 200


training_data, label = sampling(words, vocabulary_dictionary, window)
# print(training_data)
#In training data, each sample has 3 digits [61, 57, 0] as the window size is 3.
#61 is index of 'long', 57 is of ago and 0 is , 
#[[61, 57, 0], [57, 0, 1], [0, 1, 18], [1, 18, 28], [18, 28, 6], .... , [4, 20, 106], [20, 106, 48]]
# print(label)
#the next word for [61, 57, 0] is the': 1
#[1, 18, 28, 6, 24, ... , 48, 2] 


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Build a LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables with default values
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for i in range(iteration):
        last_batch = len(training_data) % batch_size
        training_steps = int(len(training_data) / batch_size) + 1
        for step in range(training_steps):
            X_batch = training_data[(step * batch_size) :((step + 1) * batch_size)]
            Y_batch = label[(step * batch_size) :((step + 1) * batch_size)]
            #X_batch is the 20 rows of the training_data in each step
            #same for Y_batch
            Y_batch_encoded = []
            for x in X_batch:
                #for each row in batch size of 20
                on_hot_vector = np.zeros([num_classes], dtype=float)
                on_hot_vector[x] = 1.0
                Y_batch_encoded = np.concatenate((Y_batch_encoded,on_hot_vector))
                
            if len(X_batch) < batch_size:
                X_batch = np.array(X_batch)
                X_batch = X_batch.reshape(last_batch, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(last_batch, num_classes)
            else:
#                 print(X_batch)
#                 [[61, 57, 0], [57, 0, 1], [0, 1, 18], [1, 18, 28], [18, 28, 6], .... , [4, 20, 106], [20, 106, 48]]
                #It will be size of 20
                X_batch = np.array(X_batch)
#                 print(X_batch)
#                 after converting it into numpy array it will be same but the structure will look like followings
#                 [[61, 57, 0] 
#                  [57, 0, 1] 
#                  [0, 1, 18] 
#                  [1, 18, 28] 
#                  [18, 28, 6]
#                   .... 
#                  [4, 20, 106]
#                  [20, 106, 48]]
                X_batch = X_batch.reshape(batch_size, timesteps, num_input)
                Y_batch_encoded = np.array(Y_batch_encoded)
                Y_batch_encoded = Y_batch_encoded.reshape(batch_size, num_classes)
            
            
            _, acc, loss, onehot_pred = sess.run([train_op, accuracy, loss_op, logits], feed_dict={X: X_batch, Y: Y_batch_encoded})
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.2f}".format(acc * 100))
