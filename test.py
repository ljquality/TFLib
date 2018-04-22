import tensorflow as tf

v = tf.Variable([[0,1,2],[1,2,3]])
w = tf.Variable([[2,3,4],[0,0,0]])
# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([1 ,1 , 5 ,1]))
sess = tf.Session()

op1 = tf.initialize_all_variables()
sess.run(op1)



#op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
#re = sess.run(op2)
#print(re)
k = tf.shape(v)[1]
optest = tf.add(v,w)

print(sess.run(k))
sess.close()