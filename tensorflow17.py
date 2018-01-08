#Tensorflow bi-directional Seq2Seq(LSTM)
#https://www.youtube.com/watch?v=ElmBrKyMXxs

import time

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from utilsSeq2SeqDate import create_date

def batch_data(x_batch, y_batch, batch_size):
    shuffle = np.random.permutation(len(x_batch))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x_batch = x_batch[shuffle]
    y_batch = y_batch[shuffle]
    while start + batch_size <= len(x_batch):
        yield x_batch[start:start+batch_size], y_batch[start:start+batch_size]
        start += batch_size
    
    return x_batch,y_batch


data = [create_date() for _ in range(50000)]

#The generated data looks like following
# print(data[:5])
#[('7 07 13', '2013-07-07'), ('30 JULY 1977', '1977-07-30'), ('Tuesday, 14 September 1971', '1971-09-14'),...]

x = [' '.join(x.split(" ")[-1::-1]) for x, y in data]
y = [y for x, y in data]

#x = ['7 07 13', '30 JULY 1977', 'Tuesday, 14 September 1971', '18 09 88', '31, Aug 1986', '10/03/1985', 'Sunday, 1 July 1979',..]
#after reverse each row in x
#x = ['13 07 7', '1977 JULY 30', '1971 September 14 Tuesday,', '88 09 18', '1986 Aug 31,', '10/03/1985', '1979 July 1 Sunday,', ..]
tokenizerX = Tokenizer()
tokenizerX.fit_on_texts(x)#Fitting X is complete.
id_to_wordX = {value:key for key,value in tokenizerX.word_index.items()}
# print(sorted(tokenizerX.word_index.items(), key=operator.itemgetter(1)))
# [('may', 1), ('10', 2), ('12', 3), ('11', 4), ('august', 5), ('january', 6), ('july', 7), ('march', 8), ('april', 9), .. ]
x_seq = tokenizerX.texts_to_sequences(x)
#[[18, 40, 50], [84, 7, 33], [91, 12, 19, 118], [126, 45, 25], [93, 37, 63], [2, 42, 105], ...]
x_pad = pad_sequences(x_seq,padding='pre') #<PAD> inbuilt in keras
# [[  0  18  40  50]
#  [  0  84   7  33]
#  [ 91  12  19 118]
#  ..., 
#  [ 66   8  53 114]
#  [  0  67  10   3]
#  [  0 110  12  55]]
#<PAD> is 0


#y = ['2013-07-07', '1977-07-30', '1971-09-14', '1988-09-18', '1986-08-31', ..]
y = ["<GO>-" + s + "-<EOS>" for s in y]
#y = ['<GO>-2013-07-07-<EOS>', '<GO>-1977-07-30-<EOS>', '<GO>-1971-09-14-<EOS>', '<GO>-1988-09-18-<EOS>', ..]


tokenizerY = Tokenizer()
tokenizerY.fit_on_texts(y)#Fitting X is complete.
id_to_wordY = {value:key for key,value in tokenizerY.word_index.items()}
#tokenizerY.word_index = {'1999': 51, '23': 21, '09': 9, '17': 20, '05': 5, '1998': 35, '1981': 67, '28': 13, '1989': 71, '02': 12,
# print(sorted(tokenizerY.word_index.items(), key=operator.itemgetter(1))) #total 80 sequences
# [('go', 1), ('eos', 2), ('07', 3), ('10', 4), ('08', 5), ('03', 6), ('05', 7), ('12', 8), ('01', 9), .. ]
y_seq = tokenizerY.texts_to_sequences(y)
#[[1, 61, 3, 3, 2], [1, 57, 3, 32, 2], [1, 68, 11, 26, 2], [1, 58, 11, 20, 2], ..]
y_pad = pad_sequences(y_seq,padding='post') #here, it does not make sense because every row of Y is of same length
# [[ 1 61  3  3  2]
#  [ 1 57  3 32  2]
#  [ 1 68 11 26  2]
#  ..., 
#  [ 1 48  6 12  2]..]

input_to_encoder = tf.placeholder(tf.int32, (None, None), 'inputs')
input_to_decoder = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

embed_size = 10
#Creating embedding matrix with random number which has values from -1.0 to 1.0
input_embedding_to_encoder = tf.Variable(tf.random_uniform((len(x_pad), embed_size), -1.0, 1.0), name='enc_embedding')
date_input_embed_to_encoder = tf.nn.embedding_lookup(input_embedding_to_encoder, input_to_encoder)

input_embedding_to_decoder = tf.Variable(tf.random_uniform((len(y_pad), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed_to_decoder = tf.nn.embedding_lookup(input_embedding_to_decoder, input_to_decoder)

nodes = 32 #number of nodes in LSTM cell
with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed_to_encoder, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
#     dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_input_embed_to_decoder, initial_state=last_state)
    ((encoder_fw_outputs,encoder_bw_outputs),(encoder_fw_final_state,encoder_bw_final_state)) = (
                                                tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_dec,
                                                                                cell_bw=lstm_dec,
                                                                                inputs=date_input_embed_to_decoder,
                                                                                dtype=tf.float32, time_major=True)
                                                                                )
  
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

batch_size = 128
#connect outputs to 
logits = tf.contrib.layers.fully_connected(encoder_outputs, num_outputs=len(y_pad), activation_fn=None) 
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, len(y_pad[0])]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


X_train, X_test, y_train, y_test = train_test_split(x_pad, y_pad, test_size=0.33, random_state=42)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch_train, target_batch_train) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits_train = sess.run([optimizer, loss, logits],
                                    feed_dict = {input_to_encoder: source_batch_train, 
                                                 input_to_decoder: target_batch_train,
                                                 targets: target_batch_train})
    
    for i,val in enumerate(batch_logits_train):
        print(batch_logits_train[i].argmax(axis=-1), target_batch_train[i])
        print('Predicted:- ')
        print([str(id_to_wordY.get(x)) for x in batch_logits_train[i].argmax(axis=-1)])
        print('Actual:- ')
        print([str(id_to_wordY.get(x)) for x in target_batch_train[i]])      
   
    accuracy = np.mean(batch_logits_train.argmax(axis=-1) == target_batch_train)
    print('Training Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, accuracy, time.time() - start_time))
    
    
    
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch_test, target_batch_test) in enumerate(batch_data(X_test, y_test, batch_size)):
        _, batch_loss, batch_logits_test = sess.run([optimizer, loss, logits],
                                    feed_dict = {input_to_encoder: source_batch_test, 
                                                 input_to_decoder: target_batch_test,
                                                 targets: target_batch_test})
    
    for i,val in enumerate(batch_logits_test):
        print(batch_logits_test[i].argmax(axis=-1), target_batch_test[i])
        print('Predicted:- ')
        print([str(id_to_wordY.get(x)) for x in batch_logits_test[i].argmax(axis=-1)])
        print('Actual:- ')
        print([str(id_to_wordY.get(x)) for x in target_batch_test[i]]) 
    accuracy = np.mean(batch_logits_test.argmax(axis=-1) == target_batch_test)
    print('Testing Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, accuracy, time.time() - start_time))