# -*- coding: utf-8 -*-

import numpy as np
import os
import time
from func_pfm import write_pfm, read_pfm
from func_makeinput import make_epiinput
from func_makeinput import make_input
from model import define_AttMLFNet
from scipy import misc

#import cv2

if __name__ == '__main__':

    dir_output = 'result'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # GPU setting ( gtx 1080ti - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    
    dir_LFimages = ['Dataset/hci_dataset/stratified/backgammon', 'Dataset/hci_dataset/stratified/dots',
                    'Dataset/hci_dataset/stratified/pyramids', 'Dataset/hci_dataset/stratified/stripes',
                    'Dataset/hci_dataset/training/boxes', 'Dataset/hci_dataset/training/cotton',
                    'Dataset/hci_dataset/training/dino', 'Dataset/hci_dataset/training/sideboard']
    '''
    dir_LFimages = ['Dataset/hci_dataset/test/bedroom', 'Dataset/hci_dataset/test/bicycle',
                    'Dataset/hci_dataset/test/herbs', 'Dataset/hci_dataset/test/origami']'''
    image_w = 512
    image_h = 512
    
    AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # number of views ( 0~8 for 9x9 )

    path_weight = dir_output + '/iter121.hdf5'

    img_scale = 1  # 1 for small_baseline(default) <3.5px,
    # 0.5 for large_baseline images   <  7px

    img_scale_inv = int(1 / img_scale)

    model_learning_rate = 0.0001
    model_512 = define_AttMLFNet(round(img_scale * image_h),
                                round(img_scale * image_w),
                                AngualrViews,
                                model_learning_rate)

    ''' Model Initialization '''

    model_512.load_weights(path_weight)
    dum_sz = model_512.input_shape[0]
    dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    tmp_list = []
    for i in range(36):
        tmp_list.append(dum)
    dummy = model_512.predict(tmp_list, batch_size=1)

    """  Depth Estimation  """
    for image_path in dir_LFimages:
        val_list = make_input(image_path, image_h, image_w, AngualrViews)

        start = time.clock()
        # predict
        val_output_tmp = model_512.predict(val_list, batch_size=1)

        runtime = time.clock() - start
        print("runtime: %.5f(s)" % runtime)

        # save results
        write_pfm(val_output_tmp[0, :, :], dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        misc.toimage(val_output_tmp[0, :, :]).save(dir_output + '/%s.png' % (image_path.split('/')[-1]))

    """ Calculate error for pre-trained model """
    output_stack = []
    gt_stack = []
    for image_path in dir_LFimages:
        output = read_pfm(dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        gt = read_pfm(image_path + '/gt_disp_lowres.pfm')
        gt_490 = gt[15:-15, 15:-15]
        output_stack.append(output)
        gt_stack.append(gt_490)

    output = np.stack(output_stack, 0)
    gt = np.stack(gt_stack, 0)

    output = output[:, 15:-15, 15:-15]

    train_diff = np.abs(output - gt)
    train_bp007 = (train_diff >= 0.07)
    train_bp003 = (train_diff >= 0.03)
    train_bp001 = (train_diff >= 0.01)

    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    training_bad007_pixel_ratio = 100 * np.average(train_bp007)
    training_bad003_pixel_ratio = 100 * np.average(train_bp003)
    training_bad001_pixel_ratio = 100 * np.average(train_bp001)
    print('Pre-trained Model average MSE*100 = %f' % training_mean_squared_error_x100)
    print('Pre-trained Model average Badpix0.07 = %f' % training_bad007_pixel_ratio)
    print('Pre-trained Model average Badpix0.03 = %f' % training_bad003_pixel_ratio)
    print('Pre-trained Model average Badpix0.01 = %f' % training_bad001_pixel_ratio)
    print('-----------------------------')
    