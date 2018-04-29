import ConvolutionNeutronNet as cnn
import ReadData as rd
import tensorflow as tf
import time

def output_reshape(n):
    l = len(n)
    output = [0.]*10*l
    for i in range(l):
        output[i*10 + n[i]] = 1.
    output = tf.reshape(output, [l, 1, 1, 10])
    return output

def input_reshape(mat):
    l = len(mat)
    return tf.reshape(mat, [l, 28, 28, 1])

def output_to_num(output):
    l = len(output)
    index = [0]*l
    for case in range(l):
        for i in range(1, 10):
            if output[case][0][0][index[case]] < output[case][0][0][i]:
                index[case] = i
        return index

layer1 = cnn.CNNNetLayer([500, 28, 28, 1])
layer1.set_filter([2, 2, 1, 6])

layer2 = cnn.CNNNetLayer([500, 14, 14, 6])
layer2.set_filter([2, 2, 6, 16])


layer3 = cnn.CNNNetLayer([500, 7, 7, 16])
layer3.set_filter([2, 2, 16, 20])


layer4 = cnn.CNNNetLayer([500, 4, 4, 20])
layer4.set_filter([3, 3, 20, 120])
layer4.set_padding('VALID')
layer4.set_if_pooling(False)


layer5 = cnn.CNNNetLayer([500, 1, 1, 120])
layer5.set_filter([1, 1, 120, 84])
layer5.set_padding('VALID')
layer5.set_if_pooling(False)

layer6 = cnn.CNNNetLayer([500, 1, 1, 84])
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
for c in range(3000):
    my_data = my_rd.read_records()

    print('read data over')
    loss = 0.
    #for i in range(300):
    my_cnn.load_data(input_reshape(my_data[0]))
    my_cnn.load_true_output_data(output_reshape(my_data[1]))
    my_cnn.calculate()
    loss = tf.add(loss, my_cnn.get_error_benchmark())
    loss = tf.div(loss, 500)

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(10):
        #pre = time.time()
        _, curloss = sess.run([train_step, loss])
        print(curloss)
        #later = time.time()
        #print (later - pre)
        if i%100==0:
            print(i)
    print('c:')
    print(c)


saver = tf.train.Saver()
saver.save(sess, 'variable_save')

sum = 0
for i in range(20):
    my_data = my_rd.read_records()
    my_cnn.load_data(input_reshape(my_data[0]))
    my_cnn.load_true_output_data(output_reshape(my_data[1]))
    result = my_cnn.calculate()
    #print(str(output_to_num(sess.run(result)))+':'+str(my_data[i][1]))
    output = output_to_num(sess.run(result))
    for j in range(500):
        if output[j] != my_data[1][j]:
            sum += 1
    #dis = tf.reduce_sum(tf.square(tf.sub(output_to_num(sess.run(result)), my_data[1])))
    #sum += dis
print(sum)
my_rd.close()
sess.close()
