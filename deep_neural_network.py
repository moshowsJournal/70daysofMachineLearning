#import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#mnist = tf.keras.datasets.mnist
#mnist = tf.keras.datasets.mnist.load_data()
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

#height x width
#X = tf.placeholder('float',[None,784])
#y = tf.placeholder('float',[None,10])

  # Create the model
X = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    # (input_data * weights) + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,
                                                              n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,
                                                              n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,
                                                            n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) ,
                hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']) ,
                hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']) ,
                hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output =    tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    return output

def train_neural_network(x,y):
    #prediction = neural_network_model(x)
    """cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for  epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _,c = optimizer.run(feed_dict = {x:x,y:y})
                epoch_loss += c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images,
                                         y:mnist.test.labels}))"""
    prediction = neural_network_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
      #sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_variables())
      hm_epochs = 10
      for i in range(hm_epochs*40):
        batch = mnist.train.next_batch(100)
        if i % hm_epochs == 0:
          test_batch_size = 10000
          batch_num = int(mnist.train.num_examples / test_batch_size)
          train_loss = 0
      
          for j in range(batch_num):
              train_loss += cross_entropy.eval(feed_dict={x:mnist.train.images[test_batch_size*j:test_batch_size*(j+1), :],y: mnist.train.labels[test_batch_size*j:test_batch_size*(j+1), :]})
            
        train_loss /= batch_num

        test_err = 1-accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print('epoch %d, training cost %g, test error %g ' % (i/hm_epochs, train_loss, test_err))
      train_step.run(feed_dict={x: batch[0], y: batch[1]}) 
        
train_neural_network(X,y)