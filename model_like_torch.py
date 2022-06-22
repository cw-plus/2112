# coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from cfg.db_config import cfg

import lib.networks.resnet.resnet_v1 as resnet_v1
import lib.networks.resnet.resnet_v1_tiny as resnet_v1_tiny
import lib.networks.mobilenet_v3 as mv3
from lib.quant_utils import quant_op


def unpool(inputs, ratio=2, name=None):
    return tf.image.resize_nearest_neighbor(inputs, size=[tf.shape(inputs)[1] * ratio,
                                                          tf.shape(inputs)[2] * ratio], name=name)


def _batch_normalization_layer(inputs, momentum=0.9, epsilon=1e-5, is_training=True, name='bn', reuse=None, fused=True):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse,
                                         fused=fused)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding,  # ('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
        channels[i] /= 255.

    return tf.concat(axis=3, values=channels)


def backbone(input, weight_decay, is_training, backbone_name=cfg.BACKBONE):
    # ['resnet_v1_50', 'resnet_v1_18', 'resnet_v2_50', 'resnet_v2_18', 'mobilenet_v2', 'mobilenet_v3']

    if backbone_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(input, is_training=is_training, scope=backbone_name)
        return logits, end_points
    elif backbone_name == 'resnet_v1_18':
        with slim.arg_scope(resnet_v1_tiny.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1_tiny.resnet_v1_18(input, is_training=is_training, scope=backbone_name)
        return logits, end_points
    elif backbone_name == 'mv3_large_prune':
        end_points = mv3.mobilenet_v3_large_prune(input, is_training=is_training)
        return None, end_points
    else:
        print('{} is error backbone name, not support!'.format(backbone_name))
        assert 0


def model(images, weight_decay=1e-5, is_training=True, quant=False):
    """
    resnet-50
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    """
    # 保证全量化模型推理
    if is_training:
        images = mean_image_subtraction(images)

    logits, end_points = backbone(images, weight_decay, is_training)

    with tf.variable_scope('decoder'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            batch_norm_params = {'decay': cfg["TRAIN"]["MOVING_AVERAGE_DECAY"],
                                 'epsilon': 1e-5,
                                 'scale': True,
                                 'is_training': is_training}

            if logits is None:
                # for k, v in end_points.items():
                #     print(k, v.shape)

                # mv3
                f = [end_points['bneck5'], end_points['bneck9'],
                     end_points['bneck11'], end_points['bneck_final']]
                c2, c3, c4, c5 = f
                inner_channels = 80

                # (1，20，20，320)->（1,20，20，80）
                in5 = _conv2d_layer(c5, filters_num=inner_channels, kernel_size=(1, 1), name="in5",
                                    use_bias=False, strides=1, padding="SAME")
                # 结论，不加relu会导致bn量化节点失败，先使用这个
                # 原网络没有,可以加一个激活函数或者identity
                in5 = tf.identity(in5)

                # (1,40,40,48)->(1,40,40,80)
                in4 = _conv2d_layer(c4, filters_num=inner_channels, kernel_size=(1, 1), name="in4",
                                    use_bias=False, strides=1, padding="SAME")
                in4 = tf.identity(in4)

                # (1,80,80,32)->(1,80,80,80)
                in3 = _conv2d_layer(c3, filters_num=inner_channels, kernel_size=(1, 1), name="in3",
                                    use_bias=False, strides=1, padding="SAME")
                in3 = tf.identity(in3)

                # (1,160,160,16)->(1,160,160,80)
                in2 = _conv2d_layer(c2, filters_num=inner_channels, kernel_size=(1, 1), name="in2",
                                    use_bias=False, strides=1, padding="SAME")
                in2 = tf.identity(in2)

                # 分辨率 20->40
                out4 = unpool(in5, name="out4") + in4  # 1/16
                # 分辨率 40->80
                out3 = unpool(out4, name="out3") + in3  # 1/8
                # 分辨率 80->160
                out2 = unpool(out3, name="out2") + in2  # 1/4

                with tf.variable_scope('concat_branch'):
                    # 1,20,20,80-> 1,20,20,20
                    in5_ = tf.pad(in5, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    p5 = _conv2d_layer(in5_, filters_num=inner_channels // 4,
                                       kernel_size=(3, 3), name="p5",
                                       use_bias=False, strides=1, padding="VALID")
                    # 结论，不加relu会导致bn量化节点失败，先使用这个
                    # 原网络没有,可以加一个激活函数或者identity
                    p5 = tf.identity(p5)

                    # 1,20,20,20-> 1,40,40,20
                    p5 = unpool(p5, ratio=8, name="p5")

                    # 1,40,40,80-> 1,40,40,20
                    out4_ = tf.pad(out4, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    p4 = _conv2d_layer(out4_, filters_num=inner_channels // 4,
                                       kernel_size=(3, 3), name="p4",
                                       use_bias=False, strides=1, padding="VALID")
                    p4 = tf.identity(p4)

                    # 1,40,40,20-> 1,80,80,20
                    p4 = unpool(p4, ratio=4, name="p4")

                    # 1,80,80,80-> 1,80,80,20
                    out3_ = tf.pad(out3, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    p3 = _conv2d_layer(out3_, filters_num=inner_channels // 4,
                                       kernel_size=(3, 3), name="p3",
                                       use_bias=False, strides=1, padding="VALID")
                    p3 = tf.identity(p3)

                    # 1,80,80,20-> 1,160,160,20
                    p3 = unpool(p3, ratio=2, name="p3")

                    # 1,160,160,80-> 1,40,40,20
                    out2_ = tf.pad(out2, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    p2 = _conv2d_layer(out2_, filters_num=inner_channels // 4,
                                       kernel_size=(3, 3), name="p2",
                                       use_bias=False, strides=1, padding="VALID")
                    p2 = tf.identity(p2)

                    fuse = tf.concat([p5, p4, p3, p2], axis=-1)
                    if quant:
                        fuse = quant_op(fuse, is_training=is_training)

                with tf.variable_scope('binarize'):
                    fuse_ = tf.pad(fuse, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    b_conv = _conv2d_layer(fuse_, filters_num=20,
                                           kernel_size=3, use_bias=False, name='fuse', padding="VALID")
                    b_conv = _batch_normalization_layer(b_conv,
                                                        momentum=0.9, epsilon=1e-5, is_training=is_training)
                    b_conv = tf.nn.relu(b_conv)

                    b_conv = tf.layers.conv2d_transpose(b_conv,
                                                        filters=20,
                                                        kernel_size=2,
                                                        strides=(2, 2),
                                                        padding='valid',
                                                        data_format='channels_last',
                                                        activation=None,
                                                        use_bias=False,
                                                        )
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.contrib.layers.bias_add(b_conv)

                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = _batch_normalization_layer(b_conv, is_training=is_training,fused=False,
                                                        momentum=0.9, epsilon=1e-5, name='binary_bn1')

                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.nn.relu(b_conv)
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.layers.conv2d_transpose(b_conv,
                                                        filters=20,
                                                        kernel_size=2,
                                                        strides=(2, 2),
                                                        padding='valid',
                                                        data_format='channels_last',
                                                        activation=None,
                                                        use_bias=False,
                                                        )
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)
                    b_conv = tf.contrib.layers.bias_add(b_conv)

                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = _batch_normalization_layer(b_conv, is_training=is_training, fused=False,
                                                        momentum=0.9, epsilon=1e-5, name='binary_bn2')
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.nn.relu(b_conv)
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    binarize_map = tf.layers.conv2d_transpose(b_conv,
                                                              filters=1,
                                                              kernel_size=2,
                                                              strides=(2, 2),
                                                              padding='valid',
                                                              data_format='channels_last',
                                                              activation=None,
                                                              use_bias=False,
                                                              )
                    if quant:
                        binarize_map = quant_op(binarize_map, is_training=is_training)

                    binarize_map = tf.contrib.layers.bias_add(binarize_map)

                    if quant:
                        binarize_map = quant_op(binarize_map, is_training=is_training)

                    binarize_map = tf.nn.sigmoid(binarize_map)

                with tf.variable_scope('threshold'):
                    fuse_ = tf.pad(fuse, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                    b_conv = _conv2d_layer(fuse_, filters_num=20, padding="VALID",
                                           kernel_size=3, use_bias=False, name='fuse')
                    b_conv = _batch_normalization_layer(b_conv,
                                                        momentum=0.9, epsilon=1e-5, is_training=is_training)
                    b_conv = tf.nn.relu(b_conv)

                    b_conv = tf.layers.conv2d_transpose(b_conv,
                                                        filters=20,
                                                        kernel_size=2,
                                                        strides=(2, 2),
                                                        padding='valid',
                                                        data_format='channels_last',
                                                        activation=None,
                                                        use_bias=False,
                                                        )
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.contrib.layers.bias_add(b_conv)

                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = _batch_normalization_layer(b_conv, is_training=is_training,fused=False,
                                                        momentum=0.9, epsilon=1e-5, name='thres_bn1')
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.nn.relu(b_conv)
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)
                    b_conv = tf.layers.conv2d_transpose(b_conv,
                                                        filters=20,
                                                        kernel_size=2,
                                                        strides=(2, 2),
                                                        padding='valid',
                                                        data_format='channels_last',
                                                        activation=None,
                                                        use_bias=False,
                                                        )
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.contrib.layers.bias_add(b_conv)
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = _batch_normalization_layer(b_conv, is_training=is_training,fused=False,
                                                        momentum=0.9, epsilon=1e-5, name='thres_bn2')
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)

                    b_conv = tf.nn.relu(b_conv)
                    if quant:
                        b_conv = quant_op(b_conv, is_training=is_training)
                    threshold_map = tf.layers.conv2d_transpose(b_conv,
                                                               filters=1,
                                                               kernel_size=2,
                                                               strides=(2, 2),
                                                               padding='valid',
                                                               data_format='channels_last',
                                                               activation=None,
                                                               use_bias=False,
                                                               )
                    if quant:
                        threshold_map = quant_op(threshold_map, is_training=is_training)

                    threshold_map = tf.contrib.layers.bias_add(threshold_map)

                    if quant:
                        threshold_map = quant_op(threshold_map, is_training=is_training)
                    threshold_map = tf.nn.sigmoid(threshold_map)

                with tf.variable_scope('thresh_binary_branch'):
                    thresh_binary = tf.reciprocal(1 + tf.exp(-cfg.K * (binarize_map - threshold_map)),
                                                  name='thresh_binary')

            else:
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                g = [None, None, None, None]
                h = [None, None, None, None]

                num_outputs = [None, 128, 64, 32]

                # size = K+(K-1)*(r-1)
                if cfg.ASPP_LAYER:
                    with tf.variable_scope('aspp_layer'):
                        f_32x = f[0]
                        f_32x_1 = slim.conv2d(f_32x, 256, 1)
                        f_32x_2 = slim.conv2d(f_32x, 256, 3)
                        f_32x_3 = slim.conv2d(f_32x, 256, 3, rate=3)
                        f_32x_4 = slim.conv2d(f_32x, 256, 3, rate=6)
                        aspp_32x = tf.concat([f_32x_1, f_32x_2, f_32x_3, f_32x_4], axis=-1)
                        f[0] = slim.conv2d(aspp_32x, 2048, 1)

                for i in range(len(f)):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)

                with tf.variable_scope('concat_branch'):
                    features = [g[3], h[2], h[1], h[0]]

                    concat_feature = None

                    for i, f in enumerate(features):
                        if i is 0:
                            conv_f = slim.conv2d(f, 64, 3)
                            concat_feature = conv_f
                        else:
                            up_f = slim.conv2d(f, 64, 3)
                            up_f = unpool(up_f, 2 ** i)
                            concat_feature = tf.concat([concat_feature, up_f], axis=-1)

                    final_f = slim.conv2d(concat_feature, 64, 3)

                with tf.variable_scope('binarize_branch'):
                    b_conv = slim.conv2d(final_f, 64, 3)
                    b_conv = slim.conv2d_transpose(b_conv, 64, 2, 2)
                    binarize_map = slim.conv2d_transpose(b_conv, 1, 2, 2, activation_fn=tf.nn.sigmoid)

                with tf.variable_scope('threshold_branch'):
                    b_conv = slim.conv2d(final_f, 64, 3)
                    b_conv = slim.conv2d_transpose(b_conv, 256, 2, 2)
                    threshold_map = slim.conv2d_transpose(b_conv, 1, 2, 2, activation_fn=tf.nn.sigmoid)

                with tf.variable_scope('thresh_binary_branch'):
                    thresh_binary = tf.reciprocal(1 + tf.exp(-cfg.K * (binarize_map - threshold_map)),
                                                  name='thresh_binary')

    return binarize_map, threshold_map, thresh_binary
