"""
Adapted from Huadong Liao's Capsnet tensorflow implementation
"""
import tensorflow as tf
from capsLayer import CapsLayer


epsilon = 1e-9

def capsnet(x,is_training,batch_size,output_size):
    with tf.variable_scope('Conv1_layer'):
        # Conv1, [batch_size, 20, 20, 256]
        conv1 = tf.contrib.layers.conv2d(x, num_outputs=256,
                                         kernel_size=9, stride=1,
                                         padding='VALID')
        assert conv1.get_shape() == [batch_size, 20, 20, 256]

    # Primary Capsules layer, return [batch_size, 1152, 8, 1]
    with tf.variable_scope('PrimaryCaps_layer'):
        primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
        caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
        assert caps1.get_shape() == [batch_size, 1152, 8, 1]

    # DigitCaps layer, return [batch_size, 100, 16, 1]
    with tf.variable_scope('DigitCaps_layer'):
        digitCaps = CapsLayer(num_outputs=output_size, vec_len=16, with_routing=True, layer_type='FC')
        caps2 = digitCaps(caps1)

    out = tf.sqrt(tf.reduce_sum(tf.square(caps2),axis=2, keep_dims=True) + epsilon) # Vector length
    return out
