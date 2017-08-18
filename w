import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()
x  = ([1.14159, 1.71828,2.51800,2.8900,3.71800,3.10059])
y  = ([ 2.91828,3.14159,3.14159,3.98655,3.5076 ,3.61828])
    
_x = tf.reduce_mean(x)
_y = tf.reduce_mean(y)

sess = tf.Session()

V = tf.reduce_sum(tf.square(tf.subtract(x,_x)))
sess.run(V)

CV = tf.reduce_sum(tf.multiply(tf.subtract(x,_x),tf.subtract(y,_y)))
sess.run(CV)

c = CV/V

m = tf.subtract(_y,tf.multiply(c,_x))
sess.run(m)

y1= tf.add(tf.multiply(m,x),c)
sess.run(y1)


rmse = tf.reduce_sum(tf.sqrt(tf.divide((tf.square(tf.subtract(y,_y))),6)))
sess.run(rmse)


