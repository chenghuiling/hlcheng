import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 1e-4
keep_prob_rate = 0.7 # dropout中留下的比例
max_epoch = 2000
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1}) # 不进行dropout
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast 将bool格式转换成浮点数
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # tf.truncated_normal() 表示截断正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    #t f.constant 表示常量
    return tf.Variable(initial)

def conv2d(x, W):
    # 每一维度  滑动步长全部是 1， padding 方式选择 same
    # 用函数 tf.nn.conv2d 进行卷积操作
    outputs = tf.nn.conv2d(input=x,filters=W,strides=[1,1,1,1],padding = 'SAME')
    #stride=[1,1,1,1]中的第1维和第4维是固定的，第2，3维表示向右和向下移动的步长
    #padding='SAME' 表示全0填充，卷积后图片的大小保持不变
    return outputs

def max_pool_2x2(x):
    # 滑动步长是 2步; 池化窗口的尺度 高和宽度都是2; padding 方式选择 same
    # 使用函数  tf.nn.max_pool 进行池化
    outputs = tf.nn.max_pool2d(input=x,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')
    #ksize = [1,2,2,1]中的第1维和第4维是固定的，第2，3维表示池化窗口的大小
    #strides = [1,2,2,1]中的第1维和第4维是固定的，第2，3维表示池化窗口移动的步长
    #padding = 'SAME' 表示使用全零填充
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28*28])/255.#255
ys = tf.placeholder(tf.float32, [None, 10])#one-hot
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])#-1=None


#  卷积层 1

W_conv1 = weight_variable([7, 7, 1, 32]) # patch 7x7, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))   # 卷积 选择激活函数Relu
h_pool1 = max_pool_2x2(h_conv1)   # 池化

# 卷积层 2
W_conv2 = weight_variable([5, 5, 32, 64])     # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))  # 卷积 选择激活函数Relu
h_pool2 = max_pool_2x2(h_conv2)    # 池化

#  全连接层 1
W_fc1 = weight_variable([7*7*64, 1024]) # 全连接层1， 1024个节点 画图
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # 将h_pool2 铺平
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #矩阵乘法，relu激活
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 随机dropout

# 全连接层 2
W_fc2 = weight_variable([1024, 10]) #全连接层2， 10个节点
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 交叉熵函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) #reduction_indices=[1] 表示按照行求和
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #变量初始化

    for i in range(max_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100) # 每次训练100个样本 
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:keep_prob_rate})
        if i % 100 == 0:
            print(compute_accuracy(
                mnist.test.images[:1000], mnist.test.labels[:1000]))