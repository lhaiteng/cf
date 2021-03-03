# -*- coding: utf-8 -*-
"""
tf.train.ExponentialMovingAverage权重滑动平均更新的使用
=> 他只是创造出待apply的var_list的shadow_var，
=> 并在每次apply时，根据那一时刻的var值，更新shadow_var，并不会影响var值
=> 因此在使用模型时，才会使用滑动平均的结果。同时有人认为，只在SGD上，权重滑动平均才有比较好的效果，其他没有。
"""
import tensorflow as tf

"""测试1"""
w = tf.Variable([1., 1.])
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, [2., 2.])

with tf.control_dependencies([update]):
    # 返回一个op,这个op用来更新moving_average,i.e. shadow value
    ema_op = ema.apply([w])  # 这句和下面那句不能调换顺序
# 以 w 当作 key， 获取 shadow value 的值
ema_val = ema.average(w)  # 参数可以是list，有点蛋疼

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run(ema_val))
    print(sess.run(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
    print(sess.run(tf.moving_average_variables()))  # 返回的是 滑动平均变量- w的更新后的值 [array([7., 7.], dtype=float32)]
    print(sess.run(w))
# 创建一个时间序列 1 2 3 4
# 输出：
# 输出：
# [1.2 1.2]
# [1.5800002 1.5800002]
# [2.1220002 2.1220002]
# []
# [array([7., 7.], dtype=float32)]
# [7. 7.]


"""测试2"""
tf.reset_default_graph()
# 定义一个32位浮点数的变量，初始值位0.0
v1 = tf.Variable(name='v1', dtype=tf.float32, initial_value=0.)

# 衰减率decay，初始值位0.99
decay = 0.99

# 定义num_updates，同样，初始值位0
num_updates = tf.Variable(0, trainable=False, name='num_updates')  # apply_decay = min(0.99, (1+0)/(10+0))=0.1

# 定义滑动平均模型的类，将衰减率decay和num_updates传入。
ema = tf.train.ExponentialMovingAverage(decay=decay, num_updates=num_updates)
# # 这样设置apply_decay = min(decay, (1+num_updates)/(10+num_updates))，
# # 在前期num_updates小时，可以应用的decay较小，更新的权重变化较快

# 定义更新变量列表
update_var_list = [v1]

# 使用滑动平均模型
ema_apply = ema.apply(update_var_list)
shadow_v1 = 0.  # 按公式计算的shadow_v1

# Tensorflow会话
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())

    # 输出初始值
    print(sess.run([v1, ema.average(v1)]))  # [0.0, 0.0]
    # [0.0, 0.0]（此时 num_updates = 0 ⇒ decay = .1, ），
    # shadow_variable = variable = 0.

    # 将v1赋值为5
    sess.run(tf.assign(v1, 5))  # [5, 0]

    # 调用函数，使用滑动平均模型
    sess.run(ema_apply)  # [5, 0*0.1+5*0.9]
    shadow_v1 = shadow_v1 * 0.1 + 5 * 0.9  # 按公式计算的shadow_v1

    # 再次输出
    print(sess.run([v1, ema.average(v1)]))  # [5.0, 4.5]
    # 此时，num_updates = 0 ⇒ decay =0.1,  v1 = 5;
    # shadow_variable = 0.1 * 0 + 0.9 * 5 = 4.5 ⇒ variable

    # 将num_updates赋值为10000
    sess.run(tf.assign(num_updates, 10000))  # apply_decay=min(decay, (1+n)/(10+n))=0.99

    # 将v1赋值为10
    sess.run(tf.assign(v1, 10))  # [10, 4.5]

    # 调用函数，使用滑动平均模型
    sess.run(ema_apply)  # [10, 4.5*0.99+10*0.01]
    shadow_v1 = shadow_v1 * 0.99 + 10 * 0.01  # 按照公式计算得到的shadow_v1

    # 输出
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.555]
    # decay = 0.99,shadow_variable = 0.99 * 4.5 + .01*10 ⇒ 4.555

    # 再次使用滑动平均模型
    sess.run(ema_apply)  # [10, 4.555*0.99+10*0.01]

    # 输出
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.60945]
    # decay = 0.99，shadow_variable = .99*4.555 + .01*10 = 4.609
    for i in range(1000):
        sess.run(ema_apply)
        shadow_v1 = shadow_v1 * 0.99 + 10 * 0.01  # 按照公式计算得到的shadow_v1
        true_shadow_v1 = sess.run(ema.average(v1))  # tf计算的shadow_v1
        print(sess.run([v1, ema.average(v1)]), shadow_v1 - true_shadow_v1)

# 恢复
# num_updates  # <tf.Variable 'num_updates:0' shape=() dtype=int32_ref>
# v1  # <tf.Variable 'v1:0' shape=() dtype=float32_ref>
# ema.average_name(v1)  # 'v1/ExponentialMovingAverage'
restore_var_list = ema.variables_to_restore()
# # {'v1/ExponentialMovingAverage': <tf.Variable 'v1:0' shape=() dtype=float32_ref>,
# # 'num_updates': <tf.Variable 'num_updates:0' shape=() dtype=int32_ref>}

# saver = tf.train.Saver(restore_var_list)
# saver.restore(sess, save_path)

"""测试3"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BATCH_SIZE = 100
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001

TRAINING_STEPS = 5000

MOVING_AVERAGE_DECAY = 0.9999


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='input_x')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='input_y')

    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1=weights1, biases1=biases1, weights2=weights2, biases2=biases2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= y,labels= tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(f'After {i} training steps, validation accuracy is {validate_acc}')

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(f'after {TRAINING_STEPS} steps, test accuracy is {test_acc}')


def main(argv=None):
    train(mnist)


if __name__ == '__main__':
    main()


