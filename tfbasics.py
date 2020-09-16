import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)
sess = tf.Session()
print(sess.run(result))