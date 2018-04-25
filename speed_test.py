import ConvolutionNeutronNet as cnn
import ReadData as rd
import tensorflow as tf
import time

def output_reshape(n):
    output = [0.]*10
    output[n] = 1.
    output = tf.reshape(output, [300, 1, 1, 10])
    return output

def input_reshape(mat):
    return tf.reshape(mat, [300, 28, 28, 1])

def output_to_num(output):
    index = 0
    for i in range(1, 10):
        if output[0][0][0][index] < output[0][0][0][i]:
            index = i
    return index

layer1 = cnn.CNNNetLayer([300, 28, 28, 1])
layer1.set_filter([2, 2, 1, 6])

layer2 = cnn.CNNNetLayer([300, 14, 14, 6])
layer2.set_filter([2, 2, 6, 16])


layer3 = cnn.CNNNetLayer([300, 7, 7, 16])
layer3.set_filter([2, 2, 16, 20])


layer4 = cnn.CNNNetLayer([300, 4, 4, 20])
layer4.set_filter([3, 3, 20, 120])
layer4.set_padding('VALID')
layer4.set_if_pooling(False)


layer5 = cnn.CNNNetLayer([300, 1, 1, 120])
layer5.set_filter([1, 1, 120, 84])
layer5.set_padding('VALID')
layer5.set_if_pooling(False)

layer6 = cnn.CNNNetLayer([300, 1, 1, 84])
layer6.set_filter([1, 1, 84, 10])
layer6.set_padding('VALID')
layer6.set_if_pooling(False)


my_cnn = cnn.CNN()
my_cnn.add_net_layer(layer1)
my_cnn.add_net_layer(layer2)
my_cnn.add_net_layer(layer3)
my_cnn.add_net_layer(layer4)
my_cnn.add_net_layer(layer5)
my_cnn.add_net_layer(layer6)

my_rd = rd.Data('./data/train_file.tfrecords')
my_data = []
for i in range(6000):
    my_data.append(my_rd.read_a_record())
my_rd.close()


print('read data over')
loss = 0.
#for i in range(300):
my_cnn.load_data(input_reshape(my_data[0:300][0]))
my_cnn.load_true_output_data(output_reshape(my_data[0:300][1]))
my_cnn.calculate()
loss = tf.add(loss, my_cnn.get_error_benchmark())
loss = tf.div(loss, 300)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session(config=tf.ConfigProto(device_count={'cpu':8}))
init = tf.initialize_all_variables()
sess.run(init)
for i in range(100):
    pre = time.time()
    _, curloss = sess.run([train_step, loss])
    print(curloss)
    later = time.time()
    print (later - pre)


#for i in range(300):
#    my_cnn.load_data(input_reshape(my_data[i][0]))
#    my_cnn.load_true_output_data(output_reshape(my_data[i][1]))
#    result = my_cnn.calculate()
#    print(str(output_to_num(sess.run(result)))+':'+str(my_data[i][1]))

sess.close()
