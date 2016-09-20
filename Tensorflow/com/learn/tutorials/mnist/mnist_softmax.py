# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
#存储路径等信息
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/Users/nocml/Documents/DATA/MNIST_data', '/tmp/result/')
#载入数据，使用onehot编码
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#初始化会话，InteractiveSession()适用于交互式的操作，比如在ipython,jupyter中使用
sess = tf.InteractiveSession()

# Create the model
#申请空间，可以理解为声明一个占位符
x = tf.placeholder(tf.float32, [None, 784])
#定义变量W，用于存储权重。大小为，行：784，列：10
W = tf.Variable(tf.zeros([784, 10]))
#定义变量b,用于存储偏置。行向量。
b = tf.Variable(tf.zeros([10]))

# y = xW + b , 此处与平时我们所见的公司 y= Wx + b略有不同。本程序的y其实是个行向量，而一般我们都习惯把y写成列向量。
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
#y_与y相对应，也是行向量
y_ = tf.placeholder(tf.float32, [None, 10])
#计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))