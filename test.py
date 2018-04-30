

import ReadData as rd

test = rd.Data('./data/train_file.tfrecords')
for i in range(1000):
    my_data=test.read_records()
    print(i)
test.close()