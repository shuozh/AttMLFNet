# -*- coding: utf-8 -*-

import numpy as np

def generate_traindata_for_train(traindata_all, traindata_label, input_size, label_size, batch_size,
                                 Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15):
    """ initialize image_stack & label """
    traindata_batch = np.zeros(
        (batch_size, input_size, input_size, len(Setting02_AngualrViews), len(Setting02_AngualrViews)),
        dtype=np.float32)

    traindata_batch_label = np.zeros((batch_size, label_size, label_size))

    """ inital variable """
    crop_half1 = int(0.5 * (input_size - label_size))

    """ Generate image stacks"""
    for ii in range(0, batch_size):
        sum_diff = 0
        valid = 0

        while (sum_diff < 0.01 * input_size * input_size or valid < 1):

            """//Variable for gray conversion//"""
            rand_3color = 0.05 + np.random.rand(3)
            rand_3color = rand_3color / np.sum(rand_3color)
            R = rand_3color[0]
            G = rand_3color[1]
            B = rand_3color[2]

            """
                We use totally 16 LF images,(0 to 15) 
                Since some images(4,6,15) have a reflection region, 
                We decrease frequency of occurrence for them. 
            """
            aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                               0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                               0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

            image_id = np.random.choice(aa_arr)

            if (len(Setting02_AngualrViews) == 9):
                ix_rd = 0
                iy_rd = 0

            kk = np.random.randint(17)
            if (kk < 8):
                scale = 1
            elif (kk < 14):
                scale = 2
            elif (kk < 17):
                scale = 3

            idx_start = np.random.randint(0, 512 - scale * input_size)
            idy_start = np.random.randint(0, 512 - scale * input_size)
            valid = 1
            """
                boolmask: reflection masks for images(4,6,15)
            """
            if (image_id == 4 or 6 or 15):
                if (image_id == 4):
                    a_tmp = boolmask_img4
                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                               idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                      idy_start: idy_start + scale * input_size:scale]) > 0):
                        valid = 0
                if (image_id == 6):
                    a_tmp = boolmask_img6
                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                               idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                      idy_start: idy_start + scale * input_size:scale]) > 0):
                        valid = 0
                if (image_id == 15):
                    a_tmp = boolmask_img15
                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                               idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                      idy_start: idy_start + scale * input_size:scale]) > 0):
                        valid = 0

            if (valid > 0):

                image_center = (1 / 255) * np.squeeze(
                    R * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 0].astype('float32') +
                    G * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 1].astype('float32') +
                    B * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 2].astype('float32'))
                sum_diff = np.sum(
                    np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))

                traindata_batch[ii, :, :, :, :] = np.squeeze(
                    R * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 0].astype(
                        'float32') +
                    G * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 1].astype(
                        'float32') +
                    B * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 2].astype(
                        'float32'))

                '''
                 traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
                '''
                if (len(traindata_label.shape) == 5):
                    traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[image_id,
                                                                      idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                      idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale,
                                                                      4 + ix_rd, 4 + iy_rd]
                else:
                    traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[image_id,
                                                                      idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                      idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]


    traindata_batch = np.float32((1 / 255) * traindata_batch)

    return traindata_batch, traindata_batch_label


def data_augmentation_for_train(traindata_batch, traindata_label_batchNxN, batch_size):
    """
        For Data augmentation
        (rotation, transpose and gamma)
    """

    for batch_i in range(batch_size):
        gray_rand = 0.4 * np.random.rand() + 0.8

        traindata_batch[batch_i, :, :, :, :] = pow(traindata_batch[batch_i, :, :, :, :], gray_rand)

        """ transpose """
        transp_rand = np.random.randint(0, 2)
        if transp_rand == 1:
            traindata_batch_tmp6 = np.copy(np.rot90(np.transpose(np.squeeze(traindata_batch[batch_i, :, :, :, :]), (1, 0, 2, 3))))
            traindata_batch[batch_i, :, :, :, :] = traindata_batch_tmp6[:, :, ::-1]
            traindata_label_batchNxN_tmp6 = np.copy(np.rot90(np.transpose(traindata_label_batchNxN[batch_i, :, :], (1, 0))))
            traindata_label_batchNxN[batch_i, :, :] = traindata_label_batchNxN_tmp6

        """ rotation """
        rotation_rand = np.random.randint(0, 4)
        """ 90 """
        if rotation_rand == 1:
            traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :])))
            traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 1, (2, 3)))
            traindata_label_batchNxN_tmp6 = np.copy(np.rot90(traindata_label_batchNxN[batch_i, :, :]))
            traindata_label_batchNxN[batch_i, :, :] = traindata_label_batchNxN_tmp6
        """ 180 """
        if rotation_rand == 2:
            traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :]), 2))
            traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 2, (2, 3)))
            traindata_label_batchNxN_tmp6 = np.copy(np.rot90(traindata_label_batchNxN[batch_i, :, :], 2))
            traindata_label_batchNxN[batch_i, :, :] = traindata_label_batchNxN_tmp6
        """ 270 """
        if rotation_rand == 3:
            traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :]), 3))
            traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 3, (2, 3)))
            traindata_label_batchNxN_tmp6 = np.copy(np.rot90(traindata_label_batchNxN[batch_i, :, :], 3))
            traindata_label_batchNxN[batch_i, :, :] = traindata_label_batchNxN_tmp6

        """ gaussian noise """
        noise_rand = np.random.randint(0, 12)
        if noise_rand == 0:
            gauss = np.random.normal(0.0, np.random.uniform()*np.sqrt(0.2), (traindata_batch.shape[1], traindata_batch.shape[2], traindata_batch.shape[3], traindata_batch.shape[4]))
            traindata_batch[batch_i, :, :, :, :] = np.clip(traindata_batch[batch_i, :, :, :, :] + gauss, 0.0, 1.0)

    return traindata_batch, traindata_label_batchNxN


def generate_traindata512(traindata_all, traindata_label, Setting02_AngualrViews):

    input_size = 512
    label_size = 512
    traindata_batch = np.zeros((len(traindata_all), input_size, input_size, len(Setting02_AngualrViews), len(Setting02_AngualrViews)), dtype=np.float32)

    traindata_label_batchNxN = np.zeros((len(traindata_all), label_size, label_size))

    """ inital setting """

    crop_half1 = int(0.5 * (input_size - label_size))

    for ii in range(0, len(traindata_all)):

        R = 0.299  ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
        G = 0.587
        B = 0.114

        image_id = ii

        ix_rd = 0
        iy_rd = 0
        idx_start = 0
        idy_start = 0

        traindata_batch[ii, :, :, :, :] = np.squeeze(
            R * traindata_all[image_id:image_id + 1, idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, :, :, 0].astype('float32') +
            G * traindata_all[image_id:image_id + 1, idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, :, :, 1].astype('float32') +
            B * traindata_all[image_id:image_id + 1, idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, :, :, 2].astype('float32'))



        if (len(traindata_all) >= 12 and traindata_label.shape[-1] == 9):
            traindata_label_batchNxN[ii, :, :] = traindata_label[image_id,
                                              idx_start + crop_half1: idx_start + crop_half1 + label_size,
                                              idy_start + crop_half1: idy_start + crop_half1 + label_size,
                                              4 + ix_rd, 4 + iy_rd]
        elif (len(traindata_label.shape) == 5):
            traindata_label_batchNxN[ii, :, :] = traindata_label[image_id,
                                                 idx_start + crop_half1: idx_start + crop_half1 + label_size,
                                                 idy_start + crop_half1: idy_start + crop_half1 + label_size, 0, 0]
        else:
            traindata_label_batchNxN[ii, :, :] = traindata_label[image_id,
                                                 idx_start + crop_half1: idx_start + crop_half1 + label_size,
                                                 idy_start + crop_half1: idy_start + crop_half1 + label_size]

    traindata_batch = np.float32((1 / 255) * traindata_batch)

    traindata_batch = np.minimum(np.maximum(traindata_batch, 0), 1)

    traindata_batch_list = []
    for j in range(traindata_batch.shape[4]):
        traindata_batch_list.append(np.expand_dims(traindata_batch[:, :, :, 4, j], axis=-1))
    for i in range(traindata_batch.shape[3]):
        traindata_batch_list.append(np.expand_dims(traindata_batch[:, :, :, i, 4], axis=-1))
    for i in range(9):
        traindata_batch_list.append(np.expand_dims(traindata_batch[:, :, :, i, i], axis=-1))
    for i in range(9):
        traindata_batch_list.append(np.expand_dims(traindata_batch[:, :, :, i, 8-i], axis=-1))
    return traindata_batch_list, traindata_label_batchNxN