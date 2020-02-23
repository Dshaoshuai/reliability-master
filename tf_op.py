import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import bisect
import math

def leaky_relu(features,alpha=0.2,name=None):
    with ops.name_scope(name,"LeakRelu",[features,alpha]):
        features=ops.convert_to_tensor(features,name='features')
        alpha=ops.convert_to_tensor(alpha,name='alpha')
        return math_ops.maximum(alpha*features,features)


def glorot(shape,scope):
    with tf.variable_scope(scope):
        init_range=10
        # init_range=shape[0]
        init_range = tf.sqrt(6 / (shape[0] + shape[1]))
        # print(shape[0],shape[1])
        init=tf.random_uniform(shape=shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
        return tf.Variable(init)
        # return tf.Variable(tf.random_uniform(shape=shape,minval=-init_range,maxval=init_range,dtype=tf.float32))
        # init

def zeros(shape,scope):
    with tf.variable_scope(scope):
        init=tf.zeros(shape,dtype=tf.float32)
        return tf.Variable(init)
        # return tf.Variable(tf.zeros(shape,dtype=dtype))

def aggregate_gradients(gradients):
    ground_gradients=[np.zeros(g.shape) for g in gradients[0]]
    for gradient in gradients:
        for i in range(len(ground_gradients)):
            ground_gradients[i]+=gradient[i]
    return ground_gradients

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    # do a piece-wise linear fit baseline
    # all_cum_rewards: list of lists of cumulative rewards
    # all_wall_time:   list of lists of physical time
    assert len(all_cum_rewards) == len(all_wall_time)

    # all unique wall time
    unique_wall_time = np.unique(np.hstack(all_wall_time))

    # for find baseline value for all unique time points
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_wall_time))

    # output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)

    return baselines