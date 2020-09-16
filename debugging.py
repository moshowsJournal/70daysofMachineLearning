import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
data = np.load("test_data.npz")
trng_input = np.array(data['Input'], dtype=np.float64)
trng_output = np.array(data['Output'], dtype=np.float64)

nhl1 = 16
nhl2 = 8
n_classes = 4


x = tf.placeholder(dtype=tf.float64, shape=[len(trng_input),24])
y = tf.placeholder(dtype=tf.float64, shape=[len(trng_output),n_classes])

def NN(data):
    hl1 = {"weights":tf.Variable(tf.random_normal([24, nhl1], dtype=tf.float64)),
           "biases":tf.Variable(tf.random_normal([nhl1], dtype=tf.float64))}

    hl2 = {"weights":tf.Variable(tf.random_normal([nhl1, nhl2], dtype=tf.float64)),
           "biases":tf.Variable(tf.random_normal([nhl2], dtype=tf.float64))}

    output_layer = {"weights":tf.Variable(tf.random_normal([nhl2, n_classes], dtype=tf.float64)),
                    "biases":tf.Variable(tf.random_normal([n_classes], dtype=tf.float64))}

    l1 = tf.add(tf.matmul(data, hl1["weights"]), hl1["biases"])
    l1 = tf.nn.leaky_relu(l1, alpha=0.2)

    l2 = tf.add(tf.matmul(l1, hl2["weights"]), hl2["biases"])
    l2 = tf.nn.leaky_relu(l2, alpha=0.2)

    output = tf.add(tf.matmul(l2, output_layer["weights"]), output_layer["biases"])
##    output = tf.nn.leaky_relu(l1, alpha=0.2)

    return output

def train(x):
    prediction = NN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            print(type(trng_input))
            print(type(trng_output))
            _, c = sess.run([optimizer, cost], feed_dict={x: trng_input, y: trng_output})
            epoch_loss += c
            print(F"Epoch {epoch} completed out of {epochs}. \nloss:{epoch_loss}")

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        Eval = accuracy.eval({x:trng_input, y:trng_output})
        print(F"Accuracy:{Eval}")

train(x)