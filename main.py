import ConvolutionNeutronNet as cnn
import ReadData as rd
import tensorflow as tf


def output_reshape(n):
    output = [0.]*10
    output[n] = 1.
    output = tf.reshape(output, [1, 1, 1, 10])
    return output

layer1 = cnn.CNNNetLayer([1, 28, 28, 1])
layer1.set_filter([2, 2, 1, 6])

layer2 = cnn.CNNNetLayer([1, 14, 14, 6])
layer2.set_filter([2, 2, 6, 16])


layer3 = cnn.CNNNetLayer([1, 7, 7, 16])
layer3.set_filter([2, 2, 16, 20])


layer4 = cnn.CNNNetLayer([1, 4, 4, 20])
layer4.set_filter([4, 4, 20, 40])
layer4.set_padding('VALID')
layer4.set_if_pooling(False)


layer5 = cnn.CNNNetLayer([1, 1, 1, 40])
layer5.set_filter([1, 1, 40, 10])
layer5.set_padding('VALID')
layer5.set_if_pooling(False)


my_cnn = cnn.CNN()
my_cnn.add_net_layer(layer1)
my_cnn.add_net_layer(layer2)
my_cnn.add_net_layer(layer3)
my_cnn.add_net_layer(layer4)
my_cnn.add_net_layer(layer5)

my_rd = rd.Data('./data/train_file.tfrecords')
my_data = []
for i in range(6000):
    my_data.append(my_rd.read_a_record())
my_rd.close()


my_cnn