#!/usr/bin/env python
#coding: utf-8

import tensorflow as tf
import input_data
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
import numpy as np
import time

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def build_network(height,width):
    """
    Function：构建网络模型。

    Parameters
    ----------
        height: Mnist图像的宽。
        width: Mnist图像的宽。

    """

    x = tf.placeholder(tf.float32, [None, height, width], name='input')

    y_placeholder = tf.placeholder(tf.float32, shape=[None, 47],name='labels_placeholder')

    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv2d_valid(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def depthwiseconv2d(x, W):
        return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1,height, width,1])

    # 1 3x3  Conv-1 28,28,1 -> 28,28,32 First Convolutional Layer
    # 1x1
    W_conv1_1 = weight_variable([1, 1, 1, 16])
    b_conv1_1 = bias_variable([16])
    h_conv1_1 = tf.nn.relu(tf.layers.BatchNormalization()(conv2d(x_image, W_conv1_1) + b_conv1_1))

    # 3x3
    W_conv1_2 = weight_variable([3, 3, 16, 1])
    b_conv1_2 = bias_variable([1])
    h_conv1_2 = tf.nn.relu(tf.layers.BatchNormalization()(depthwiseconv2d(h_conv1_1, W_conv1_2) + b_conv1_2))

    # 1x1
    W_conv1_3 = weight_variable([1, 1, 16, 1])
    b_conv1_3 = bias_variable([1])
    h_conv1_3 = tf.layers.BatchNormalization()(conv2d(h_conv1_2, W_conv1_3) + b_conv1_3) + x_image

    # 2 3x3  Conv-2 28,28,32 -> 28,28,32
    # 1x1
    W_conv2_1 = weight_variable([1, 1, 1, 32])
    b_conv2_1 = bias_variable([32])
    h_conv2_1 = tf.nn.relu(tf.layers.BatchNormalization()(conv2d(h_conv1_3, W_conv2_1) + b_conv2_1))

    # 3x3
    W_conv2_2 = weight_variable([3, 3, 32, 1])
    b_conv2_2 = bias_variable([1])
    h_conv2_2 = tf.nn.relu(tf.layers.BatchNormalization()(depthwiseconv2d(h_conv2_1, W_conv2_2) + b_conv2_2))

    # 1x1
    W_conv2_3 = weight_variable([1, 1, 32, 1])
    b_conv2_3 = bias_variable([1])
    h_conv2_3 = tf.layers.BatchNormalization()(conv2d(h_conv2_2, W_conv2_3) + b_conv2_3) + h_conv1_3


    # 2 MaxPool-1 28,28,32 -> 14,14,32
    h_pool1 = max_pool_2x2(h_conv2_3)

    # 3 3x3 Conv-3 14,14,32 -> 14,14,64 Second Convolutional Layer
    # 1x1
    W_conv3_1 = weight_variable([1, 1, 1, 64])
    b_conv3_1 = bias_variable([64])
    h_conv3_1 = tf.nn.relu(tf.layers.BatchNormalization()(conv2d(h_pool1, W_conv3_1) + b_conv3_1))

    # 3x3
    W_conv3_2 = weight_variable([3, 3, 64, 1])
    b_conv3_2 = bias_variable([1])
    h_conv3_2 = tf.nn.relu(tf.layers.BatchNormalization()(depthwiseconv2d(h_conv3_1, W_conv3_2) + b_conv3_2))

    # 1x1
    W_conv3_3 = weight_variable([1, 1, 64, 1])
    b_conv3_3 = bias_variable([1])
    h_conv3_3 = tf.layers.BatchNormalization()(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3) + h_pool1

    h_pool2 = max_pool_2x2(h_conv3_3)


    # 4 3x3 Conv-4
    # 1x1
    #W_conv4_1 = weight_variable([3, 3, 1, 64])
    #b_conv4_1 = bias_variable([64])
    #h_conv4_1 = tf.nn.relu(tf.layers.BatchNormalization()(conv2d_valid(h_pool2, W_conv4_1) + b_conv4_1))

    # 3x3
    #W_conv4_2 = weight_variable([3, 3, 64, 1])
    #b_conv4_2 = bias_variable([1])
    #h_conv4_2 = tf.nn.relu(tf.layers.BatchNormalization()(depthwiseconv2d(h_conv4_1, W_conv4_2) + b_conv4_2))

    # 1x1
    #W_conv4_3 = weight_variable([1, 1, 64, 1])
    #b_conv4_3 = bias_variable([1])
    #h_conv4_3 = tf.layers.BatchNormalization()(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)


    # 4 3x3 Conv-5
    # 1x1
    W_conv5_1 = weight_variable([3, 3, 1, 128])
    b_conv5_1 = bias_variable([128])
    h_conv5_1 = tf.nn.relu(tf.layers.BatchNormalization()(conv2d_valid(h_pool2, W_conv5_1) + b_conv5_1))

    # 3x3
    W_conv5_2 = weight_variable([3, 3, 128, 1])
    b_conv5_2 = bias_variable([1])
    h_conv5_2 = tf.nn.relu(tf.layers.BatchNormalization()(depthwiseconv2d(h_conv5_1, W_conv5_2) + b_conv5_2))

    # 1x1
    W_conv5_3 = weight_variable([1, 1, 128, 1])
    b_conv5_3 = bias_variable([1])
    h_conv5_3 = tf.layers.BatchNormalization()(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)

    # 5 FC-1  Densely Connected Layer
    W_fc1 = weight_variable([5 * 5 * 1, 512])
    b_fc1 = bias_variable([512])

    h_pool2_flat = tf.reshape(h_conv5_3, [-1, 5*5*1])
    print h_pool2_flat
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print h_fc1
    # 6 Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_placeholder)

    # 7 FC-2 Readout Layer
    W_fc2 = weight_variable([512, 47])
    b_fc2 = bias_variable([47])

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # output
    sofmax_out = tf.nn.softmax(logits,name="out_softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_placeholder))

    # decayed learning rate
    # 6w/1000*10=6
    # 0.001 * 0.98^(6*30)  = 0.001 * 0.0263
    global_step = tf.Variable(0)
    initial_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=1000,
                                               decay_rate=0.98)

    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step)

    #optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    prediction_labels = tf.argmax(sofmax_out, axis=1,name="output")
    real_labels= tf.argmax(y_placeholder, axis=1)

    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
                keep_prob_placeholder = keep_prob_placeholder,
                x_placeholder= x,
                y_placeholder = y_placeholder,
                optimize = optimize,
                logits = logits,
                prediction_labels = prediction_labels,
                real_labels = real_labels,
                correct_prediction = correct_prediction,
                correct_times_in_batch = correct_times_in_batch,
                cost = cost,
                accuracy = accuracy,
    )

def train_network(graph,
                 dataset,
                 batch_size,
                 num_epochs,
                 pb_file_path,):
    """
    Function：训练网络。

    Parameters
    ----------
        graph: 一个dict,build_network函数的返回值。
        dataset: 数据集
        batch_size:
        num_epochs: 训练轮数。
        pb_file_path：要生成的pb文件的存放路径。
    """

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print "batch size:",batch_size

        # figure
        fig_acy_on_train= np.zeros([num_epochs])
        fig_loss_on_train = np.zeros([num_epochs])
        fig_acy_on_test = np.zeros([num_epochs])
        fig_loss_on_test = np.zeros([num_epochs])

        epoch_cost_time_train = np.zeros([num_epochs])
        epoch_cost_time_test = np.zeros([num_epochs])
        epoch_cost_time = np.zeros([num_epochs])

        #用于控制每epoch_delta轮在train set和test set上计算一下accuracy和cost
        #epoch_delta = 2
        epoch_delta = 1

        for epoch_index in range(num_epochs):

            #print "epoch_index", epoch_index

            epoch_start_time_train = time.time()

            #################################
            #    获取TRAIN set，开始训练网络
            #################################
            for (batch_xs,batch_ys) in dataset.train.mini_batches(batch_size):
                sess.run([graph['optimize']], feed_dict={
                    graph['x_placeholder']: batch_xs,
                    graph['y_placeholder']: batch_ys,
                    graph['keep_prob_placeholder']:0.5,
                })


            #每epoch_delta轮在train set和test set上计算一下accuracy和cost
            if epoch_index % epoch_delta  == 0:
                #################################
                #    开始在 train set上计算一下accuracy和cost
                #################################
                #记录训练集中有多少个batch
                total_batches_in_train_set = 0
                #记录在训练集中预测正确的次数
                total_correct_times_in_train_set = 0
                #记录在训练集中的总cost
                total_cost_in_train_set = 0.
                #print "len:",len(dataset.train.mini_batches(batch_size))
                for (train_batch_xs,train_batch_ys) in dataset.train.mini_batches(batch_size):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })

                    #print "total_batches_in_train_set: ", total_batches_in_train_set

                    #print "return_correct_times_in_batch:", return_correct_times_in_batch

                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })
                    #print "mean_cost_in_batch:", mean_cost_in_batch


                    total_batches_in_train_set += 1
                    #print "total_batches_in_train_set:", total_batches_in_train_set

                    total_correct_times_in_train_set += return_correct_times_in_batch
                    #print "total_correct_times_in_train_set:", total_correct_times_in_train_set

                    total_cost_in_train_set += (mean_cost_in_batch*batch_size)
                    #print "total_cost_in_train_set:", total_cost_in_train_set

                    #print "###########",total_batches_in_train_set

                epoch_end_time_train = time.time()
                epoch_cost_time_train[epoch_index] = epoch_end_time_train - epoch_start_time_train

                #################################
                # 开始在 test set上计算一下accuracy和cost
                #################################
                #记录测试集中有多少个batch
                total_batches_in_test_set = 0
                #记录在测试集中预测正确的次数
                total_correct_times_in_test_set = 0
                #记录在测试集中的总cost
                total_cost_in_test_set = 0.
                for (test_batch_xs,test_batch_ys) in dataset.test.mini_batches(batch_size):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })

                    #print "total_batches_in_test_set: ", total_batches_in_test_set

                    #print "return_correct_times_in_batch:", return_correct_times_in_batch

                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })

                    #print "mean_cost_in_batch:", mean_cost_in_batch

                    total_batches_in_test_set += 1
                    #print "total_batches_in_test_set:", total_batches_in_test_set
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    #print "total_correct_times_in_test_set:",total_correct_times_in_test_set
                    total_cost_in_test_set  += (mean_cost_in_batch*batch_size)
                    #print "total_cost_in_test_set:",total_cost_in_test_set

                epoch_end_time_test = time.time()
                epoch_cost_time_test[epoch_index] = epoch_end_time_test - epoch_end_time_train
                epoch_cost_time[epoch_index] = epoch_end_time_test - epoch_start_time_train

                ### summary and print
                acy_on_test = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)

                print('Epoch - {:2d} , acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}, time:{}'.
                      format(epoch_index, acy_on_test*100.0,total_correct_times_in_test_set,
                             total_batches_in_test_set * batch_size,total_cost_in_test_set, acy_on_train*100.0,
                             total_correct_times_in_train_set,total_batches_in_train_set * batch_size,total_cost_in_train_set, epoch_cost_time[epoch_index]))

            # epoch accuracy figure
            fig_acy_on_train[epoch_index] = acy_on_train
            fig_loss_on_train[epoch_index] = total_cost_in_train_set
            fig_acy_on_test[epoch_index] = acy_on_test
            fig_loss_on_test[epoch_index] = total_cost_in_test_set

            # 每轮训练完后就保存为pb文件
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"]) #out_softmax
            with tf.gfile.FastGFile(pb_file_path+"mv2_relu_6w_128_5X5_512_emnist_balanced"+str(epoch_index)+".pb",mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        stats_graph(sess.graph)
        print "##########################################"



    print "epoch_cost_time_train_mobilenetv2_relu_6w=", epoch_cost_time_train
    print "epoch_cost_time_test_mobilenetv2_relu_6w=", epoch_cost_time_test
    print "epoch_cost_time_mobilenetv2_relu_6w=", epoch_cost_time
    print "fig_acy_on_train_mobilenetv2_relu_6w=", fig_acy_on_train
    print "fig_acy_on_test_mobilenetv2_relu_6w=", fig_acy_on_test
    print "fig_loss_on_train_mobilenetv2_relu_6w=", fig_loss_on_train
    print "fig_loss_on_test_mobilenetv2_relu_6w=", fig_loss_on_test


def main():

    batch_size = 10 #original 20
    #num_epochs = 50
    num_epochs = 30

    #pb文件保存路径
    pb_file_path = "model/"

    build_start_time = time.time()
    g = build_network(height=28, width=28)
    build_end_time = time.time()
    print "build time:", build_end_time - build_start_time

    input_start_time = time.time()
    dataset = input_data.read_data_sets()
    input_end_time = time.time()
    print "input time", input_end_time - input_start_time

    train_start_time = time.time()
    train_network(g, dataset, batch_size, num_epochs, pb_file_path)
    train_end_time = time.time()
    print "train time", train_end_time - train_start_time

main()
