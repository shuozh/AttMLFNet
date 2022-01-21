from tensorflow.contrib.keras.api.keras.optimizers import RMSprop, Adam
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input, Activation, BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape, Conv3DTranspose, Conv3D, AveragePooling2D, Lambda, UpSampling2D, \
    UpSampling3D, GlobalAveragePooling3D
from tensorflow.contrib.keras.api.keras.layers import Dropout, BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate, add, multiply

import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import numpy as np


def convbn(input, out_planes, kernel_size, stride, dilation):
    seq = Conv2D(out_planes, kernel_size, stride, 'same', dilation_rate=dilation, data_format='channels_last',
                 use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq


def convbn_3d(input, out_planes, kernel_size, stride):
    seq = Conv3D(out_planes, kernel_size, stride, 'same', data_format='channels_last', use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq


def BasicBlock(input, planes, stride, downsample, dilation):
    conv1 = convbn(input, planes, 3, stride, dilation)
    conv1 = Activation('relu')(conv1)
    conv2 = convbn(conv1, planes, 3, 1, dilation)
    if downsample is not None:
        input = downsample

    conv2 = add([conv2, input])
    return conv2


def _make_layer(input, planes, blocks, stride, dilation):
    inplanes = 4
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = Conv2D(planes, 1, stride, 'same', data_format='channels_last', use_bias=False)(input)
        downsample = BatchNormalization()(downsample)

    layers = BasicBlock(input, planes, stride, downsample, dilation)
    for i in range(1, blocks):
        layers = BasicBlock(layers, planes, 1, None, dilation)

    return layers


def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))


def UpSampling3DBilinear(size):
    def UpSampling3DBilinear_(x, size):
        shape = K.shape(x)
        x = K.reshape(x, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        x = tf.image.resize_bilinear(x, size, align_corners=True)
        x = K.reshape(x, (shape[0], shape[1], size[0], size[1], shape[4]))
        return x

    return Lambda(lambda x: UpSampling3DBilinear_(x, size))


def feature_extraction(sz_input, sz_input2):
    i = Input(shape=(sz_input, sz_input2, 1))
    firstconv = convbn(i, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)
    firstconv = convbn(firstconv, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)

    layer1 = _make_layer(firstconv, 4, 2, 1, 1)  # (?, 32, 32, 4)
    layer2 = _make_layer(layer1, 8, 8, 1, 1)  # (?, 32, 32, 8)
    layer3 = _make_layer(layer2, 16, 2, 1, 1)  # (?, 32, 32, 16)
    layer4 = _make_layer(layer3, 16, 2, 1, 2)  # (?, 32, 32, 16)
    layer4_size = (layer4.get_shape().as_list()[1], layer4.get_shape().as_list()[2])

    branch1 = AveragePooling2D((2, 2), (2, 2), 'same', data_format='channels_last')(layer4)
    branch1 = convbn(branch1, 4, 1, 1, 1)
    branch1 = Activation('relu')(branch1)
    branch1 = UpSampling2DBilinear(layer4_size)(branch1)

    branch2 = AveragePooling2D((4, 4), (4, 4), 'same', data_format='channels_last')(layer4)
    branch2 = convbn(branch2, 4, 1, 1, 1)
    branch2 = Activation('relu')(branch2)
    branch2 = UpSampling2DBilinear(layer4_size)(branch2)

    branch3 = AveragePooling2D((8, 8), (8, 8), 'same', data_format='channels_last')(layer4)
    branch3 = convbn(branch3, 4, 1, 1, 1)
    branch3 = Activation('relu')(branch3)
    branch3 = UpSampling2DBilinear(layer4_size)(branch3)

    branch4 = AveragePooling2D((16, 16), (16, 16), 'same', data_format='channels_last')(layer4)
    branch4 = convbn(branch4, 4, 1, 1, 1)
    branch4 = Activation('relu')(branch4)
    branch4 = UpSampling2DBilinear(layer4_size)(branch4)

    output_feature = concatenate([layer2, layer4, branch4, branch3, branch2, branch1], )
    lastconv = convbn(output_feature, 16, 3, 1, 1)
    lastconv = Activation('relu')(lastconv)
    lastconv = Conv2D(4, 1, (1, 1), 'same', data_format='channels_last', use_bias=False)(lastconv)
    print(lastconv.get_shape())
    model = Model(inputs=[i], outputs=[lastconv])

    return model
def _get_h_CostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    for d in range(-4, 5):
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                v = v + 4
                tensor = tf.contrib.image.translate(inputs[i], [d * (u - 4), d * (v - 4)], 'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume, (shape[0], 9, shape[1], shape[2], 4 * 9))
    return cost_volume
def _get_v_CostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    for d in range(-4, 5):
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                v = v + 4
                tensor = tf.contrib.image.translate(inputs[i], [d * (v - 4), d * (u - 4)], 'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume, (shape[0], 9, shape[1], shape[2], 4 * 9))
    return cost_volume
def _get_45_CostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    for d in range(-4, 5):
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                v = v + i
                tensor = tf.contrib.image.translate(inputs[i], [d * (u - 4), d * (v - 4)], 'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume, (shape[0], 9, shape[1], shape[2], 4 * 9))
    return cost_volume
def _get_135_CostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    for d in range(-4, 5):
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                v = v + i
                u = 8 - u
                tensor = tf.contrib.image.translate(inputs[i], [d * (u - 4), d * (v - 4)], 'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume, (shape[0], 9, shape[1], shape[2], 4 * 9))
    return cost_volume

def basic(cost_volume):
    feature = 2 * 75
    dres0 = convbn_3d(cost_volume, feature, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, feature, 3, 1)
    cost0 = Activation('relu')(dres0)

    dres1 = convbn_3d(cost0, feature, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(dres1, feature, 3, 1)
    cost0 = add([dres1, cost0])

    dres4 = convbn_3d(cost0, feature, 3, 1)
    dres4 = Activation('relu')(dres4)
    dres4 = convbn_3d(dres4, feature, 3, 1)
    cost0 = add([dres4, cost0])

    classify = convbn_3d(cost0, feature, 3, 1)
    classify = Activation('relu')(classify)
    cost = Conv3D(1, 3, 1, 'same', data_format='channels_last', use_bias=False)(classify)

    return cost
def to_3d_h(cost_volume_h):
    feature = 4 * 9
    channel_h = GlobalAveragePooling3D(data_format='channels_last')(cost_volume_h)
    channel_h = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(channel_h)
    channel_h = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(channel_h)
    channel_h = Activation('relu')(channel_h)
    channel_h = Conv3D(3, 1, 1, 'same', data_format='channels_last')(channel_h)
    channel_h = Activation('sigmoid')(channel_h)
    channel_h = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(channel_h)
    channel_h = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 9)))(channel_h)
    channel_h = Lambda(lambda y: K.repeat_elements(y, 4, -1))(channel_h)
    cv_h_tmp = multiply([channel_h, cost_volume_h])
    cv_h_tmp = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(cv_h_tmp)
    cv_h_tmp = Activation('relu')(cv_h_tmp)
    cv_h_tmp = Conv3D(3, 1, 1, 'same', data_format='channels_last')(cv_h_tmp)
    cv_h_tmp = Activation('sigmoid')(cv_h_tmp)
    attention_h = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(cv_h_tmp)
    attention_h = Lambda(lambda y: K.repeat_elements(y, 4, -1))(attention_h)
    cv_h_multi = multiply([attention_h, cost_volume_h])
    dres0 = convbn_3d(cv_h_multi, feature, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(cv_h_multi, feature/2, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(cv_h_multi, feature/2, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(cv_h_multi, feature/4, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, 1, 3, 1)
    cost0 = Activation('relu')(dres0)
    cost0 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost0)
    return cost0, cv_h_multi

def to_3d_v(cost_volume_v):
    feature = 4 * 9
    channel_v = GlobalAveragePooling3D(data_format='channels_last')(cost_volume_v)
    channel_v = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(channel_v)
    channel_v = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(channel_v)
    channel_v = Activation('relu')(channel_v)
    channel_v = Conv3D(3, 1, 1, 'same', data_format='channels_last')(channel_v)
    channel_v = Activation('sigmoid')(channel_v)
    channel_v = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(channel_v)
    channel_v = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 9)))(channel_v)
    channel_v = Lambda(lambda y: K.repeat_elements(y, 4, -1))(channel_v)
    cv_v_tmp = multiply([channel_v, cost_volume_v])
    cv_v_tmp = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(cv_v_tmp)
    cv_v_tmp = Activation('relu')(cv_v_tmp)
    cv_v_tmp = Conv3D(3, 1, 1, 'same', data_format='channels_last')(cv_v_tmp)
    cv_v_tmp = Activation('sigmoid')(cv_v_tmp)
    attention_v = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(cv_v_tmp)
    attention_v = Lambda(lambda y: K.repeat_elements(y, 4, -1))(attention_v)
    cv_v_multi = multiply([attention_v, cost_volume_v])
    dres1 = convbn_3d(cv_v_multi, feature, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(cv_v_multi, feature/2, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(cv_v_multi, feature/2, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(cv_v_multi, feature/4, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(dres1, 1, 3, 1)
    cost1 = Activation('relu')(dres1)
    cost1 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost1)
    return cost1, cv_v_multi
def to_3d_45(cost_volume_45):
    feature = 4 * 9
    channel_45 = GlobalAveragePooling3D(data_format='channels_last')(cost_volume_45)
    channel_45 = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(channel_45)
    channel_45 = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(channel_45)
    channel_45 = Activation('relu')(channel_45)
    channel_45 = Conv3D(3, 1, 1, 'same', data_format='channels_last')(channel_45)
    channel_45 = Activation('sigmoid')(channel_45)
    channel_45 = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(channel_45)
    channel_45 = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 9)))(channel_45)
    channel_45 = Lambda(lambda y: K.repeat_elements(y, 4, -1))(channel_45)
    cv_45_tmp = multiply([channel_45, cost_volume_45])
    cv_45_tmp = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(cv_45_tmp)
    cv_45_tmp = Activation('relu')(cv_45_tmp)
    cv_45_tmp = Conv3D(3, 1, 1, 'same', data_format='channels_last')(cv_45_tmp)
    cv_45_tmp = Activation('sigmoid')(cv_45_tmp)
    attention_45 = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(cv_45_tmp)
    attention_45 = Lambda(lambda y: K.repeat_elements(y, 4, -1))(attention_45)
    cv_45_multi = multiply([attention_45, cost_volume_45])
    dres2 = convbn_3d(cv_45_multi, feature, 3, 1)
    dres2 = Activation('relu')(dres2)
    dres2 = convbn_3d(cv_45_multi, feature/2, 3, 1)
    dres2 = Activation('relu')(dres2)
    dres2 = convbn_3d(cv_45_multi, feature/2, 3, 1)
    dres2 = Activation('relu')(dres2)
    dres2 = convbn_3d(cv_45_multi, feature/4, 3, 1)
    dres2 = Activation('relu')(dres2)
    dres2 = convbn_3d(dres2, 1, 3, 1)
    cost2 = Activation('relu')(dres2)
    cost2 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost2)
    return cost2, cv_45_multi
def to_3d_135(cost_volume_135):
    feature = 4 * 9
    channel_135 = GlobalAveragePooling3D(data_format='channels_last')(cost_volume_135)
    channel_135 = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(channel_135)
    channel_135 = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(channel_135)
    channel_135 = Activation('relu')(channel_135)
    channel_135 = Conv3D(3, 1, 1, 'same', data_format='channels_last')(channel_135)
    channel_135 = Activation('sigmoid')(channel_135)
    channel_135 = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(channel_135)
    channel_135 = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 9)))(channel_135)
    channel_135 = Lambda(lambda y: K.repeat_elements(y, 4, -1))(channel_135)
    cv_135_tmp = multiply([channel_135, cost_volume_135])
    cv_135_tmp = Conv3D(feature/2, 1, 1, 'same', data_format='channels_last')(cv_135_tmp)
    cv_135_tmp = Activation('relu')(cv_135_tmp)
    cv_135_tmp = Conv3D(3, 1, 1, 'same', data_format='channels_last')(cv_135_tmp)
    cv_135_tmp = Activation('sigmoid')(cv_135_tmp)
    attention_135 = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:1], y[:, :, :, :, 0:1], y[:, :, :, :, 0:1],y[:, :, :, :, 0:1],
                                        y[:, :, :, :, 1:2],
                                        y[:, :, :, :, 2:3], y[:, :, :, :, 2:3], y[:, :, :, :, 2:3],y[:, :, :, :, 2:3]], axis=-1))(cv_135_tmp)
    attention_135 = Lambda(lambda y: K.repeat_elements(y, 4, -1))(attention_135)
    cv_135_multi = multiply([attention_135, cost_volume_135])
    dres3 = convbn_3d(cv_135_multi, feature, 3, 1)
    dres3 = Activation('relu')(dres3)
    dres3 = convbn_3d(cv_135_multi, feature/2, 3, 1)
    dres3 = Activation('relu')(dres3)
    dres3 = convbn_3d(cv_135_multi, feature/2, 3, 1)
    dres3 = Activation('relu')(dres3)
    dres3 = convbn_3d(cv_135_multi, feature/4, 3, 1)
    dres3 = Activation('relu')(dres3)
    dres3 = convbn_3d(dres3, 1, 3, 1)
    cost3 = Activation('relu')(dres3)
    cost3 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost3)
    return cost3, cv_135_multi
def branch_attention(cost_volume_3d,cost_volume_h,cost_volume_v,cost_volume_45,cost_volume_135): 
    feature = 4 * 9
    k = 9
    label = 9
    cost1 = convbn(cost_volume_3d, 6,3,1,1)
    cost1 = Activation('relu')(cost1)
    cost1 = convbn(cost1, 4, 3, 1, 1)
    cost1 = Activation('sigmoid')(cost1)
    cost_h = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:,:,:,:1], 1), 9, 1))(cost1)
    cost_h = Lambda(lambda y: K.repeat_elements(y, feature, 4))(cost_h)
    cost_v = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:,:,:,1:2], 1), 9, 1))(cost1)
    cost_v = Lambda(lambda y: K.repeat_elements(y, feature, 4))(cost_v)
    cost_45 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:,:,:,2:3], 1), 9, 1))(cost1)
    cost_45 = Lambda(lambda y: K.repeat_elements(y, feature, 4))(cost_45)
    cost_135 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:,:,:,3:4], 1), 9, 1))(cost1)
    cost_135 = Lambda(lambda y: K.repeat_elements(y, feature, 4))(cost_135)
    return concatenate([multiply([cost_h,cost_volume_h]),multiply([cost_v,cost_volume_v]),multiply([cost_45,cost_volume_45]),multiply([cost_135,cost_volume_135])],axis=4),cost1
def spatial_attention(cost_volume):
    feature = 4 * 9
    k = 9
    label = 9
    dres0 = convbn_3d(cost_volume, feature/2, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, 1, 3, 1)
    cost0 = Activation('relu')(dres0)

    cost0 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost0)

    cost1 = convbn(cost0, label//2,(1,k),1,1)
    cost1 = Activation('relu')(cost1)
    cost1 = convbn(cost1, 1, (k, 1), 1, 1)
    cost1 = Activation('relu')(cost1)

    cost2 = convbn(cost0, label // 2, (k, 1), 1, 1)
    cost2 = Activation('relu')(cost2)
    cost2 = convbn(cost2, 1, (1, k), 1, 1)
    cost2 = Activation('relu')(cost2)

    cost = add([cost1,cost2])
    cost = Activation('sigmoid')(cost)

    cost = Lambda(lambda y: K.repeat_elements(K.expand_dims(y, 1), 9, 1))(cost)
    cost = Lambda(lambda y: K.repeat_elements(y, feature, 4))(cost)
    return multiply([cost,cost_volume])

def disparityregression(input):
    shape = K.shape(input)
    disparity_values = np.linspace(-4, 4, 9)
    x = K.constant(disparity_values, shape=[9])
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1])
    out = K.sum(multiply([input, x]), -1)
    return out

def define_AttMLFNet(sz_input, sz_input2, view_n, learning_rate):
    """ 4 branches inputs"""
    input_list = []
    for i in range(len(view_n)*4):
        input_list.append(Input(shape=(sz_input, sz_input2, 1)))


    """ 4 branches features"""
    feature_extraction_layer = feature_extraction(sz_input, sz_input2)
    feature_list = []
    for i in range(len(view_n)*4):
        feature_list.append(feature_extraction_layer(input_list[i]))
    feature_v_list = []
    feature_h_list = []
    feature_45_list = []
    feature_135_list = []
    for i in range(9):
        feature_h_list.append(feature_list[i])
    for i in range(9,18):
        feature_v_list.append(feature_list[i])
    for i in range(18,27):
        feature_45_list.append(feature_list[i])
    for i in range(27,len(feature_list)):
        feature_135_list.append(feature_list[i])
    """ cost volume """
    cv_h = Lambda(_get_h_CostVolume_)(feature_h_list)
    cv_v = Lambda(_get_v_CostVolume_)(feature_v_list)
    cv_45 = Lambda(_get_45_CostVolume_)(feature_45_list)
    cv_135 = Lambda(_get_135_CostVolume_)(feature_135_list)
    """ intra branch """
    cv_h_3d, cv_h_ca = to_3d_h(cv_h)
    cv_v_3d, cv_v_ca = to_3d_v(cv_v)
    cv_45_3d, cv_45_ca = to_3d_45(cv_45)
    cv_135_3d, cv_135_ca = to_3d_135(cv_135)
    """ inter branch """
    cv, attention_4 = branch_attention(multiply([cv_h_3d,cv_v_3d,cv_45_3d,cv_135_3d]),cv_h_ca,cv_v_ca,cv_45_ca,cv_135_ca)
    """ cost volume regression """
    cost = basic(cv)

    cost = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost)
    pred = Activation('softmax')(cost)
    pred = Lambda(disparityregression)(pred)

    model = Model(inputs = input_list, outputs=[pred])

    model.summary()

    opt = Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='mae')

    return model