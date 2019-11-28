import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from tensorflow.examples.tutorials.mnist import input_data

# input, 28x28 = 784개의 값을 가지며 n개의 이미지
X = tf.placeholder(tf.float32, [None, 784])
# input 을 이미지로 인식하기 위해 reshape을 해준다. 28*28의 이미지이며 단일색상, 개수는 n개이므로 -1
X_img = tf.reshape(X, [-1, 28, 28, 1])
# output
Y = tf.placeholder(tf.float32, [None, 10])

# layer 1
# 3*3크기의 필터, 색상은 단일, 총 32개의 필터
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))
# conv2d 를 통과해도 28*28 크기를 가짐, 대신 32개의 필터이므로 총 32개의 결과가 생김
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# max pooling을 하고 나면 스트라이드 및 패딩 설정에 의해 14*14크기의 결과가 나옴
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 2
# 이번에는 64개의 필터
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.1))
# conv2d layer를 통과시키면, [?,14,14,64] 형태를 가짐
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
# max pooling 에서 stride가 2 이므로, 결과는 7 * 7 형태를 가질
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 이후 쭉 펼친다.
L2 = tf.reshape(L2, [-1,7 * 7 * 64])

# fully-connected layer
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets("~/deep_learning_zeroToAll/", one_hot=True)
training_epochs = 15
batch_size = 100

# train
print('Learning started. It takes sometimes.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print('Learning Finished!')

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

img = cv2.imread("C:/Users/ghkde/Documents/TensorTrainingData/3.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()

img = cv2.resize(255-img, (28, 28))
test_num = img.flatten()/255.0
test_num = test_num.reshape((-1, 28, 28, 1))

print('The answer is ', model.predict_classes(test_num))
plt.show()