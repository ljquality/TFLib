import tensorflow as tf
import ReadData



t = [0]*10
t[5] = 1
sess = tf.Session()
t = tf.reshape(t, [2,1,5])

print sess.run(t)

sess.close()

