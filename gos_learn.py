from __future__ import absolute_import, division, print_function, unicode_literals
# Set Specific GPU in tensorflow
# We can check GPU information with command ```nvidia-smi```.
# If there is only one GPU avaliable, we set the os environ to "0".
# If there are two GPU avaliable and we would like to set the second GPU, 
#   we set the os environ to "1".
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import data.preprocessing as prep
from model.model import generate_weight
from model.device_model import Device_Model
from data.data_utils import load_cifar10_data, train_test_label_to_categorical

from math import floor
from sklearn.utils import shuffle
from random import randint
from tensorflow.keras import datasets
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

with tf.device('/device:GPU:0'):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
_, epo = 0, 0

def step_decay(epoch):
    epoch = _ + epo
    # initial_lrate = 1.0 # no longer needed
    drop = 0.99
    epochs_drop = 1.0
    
    lrate = 0.2 * pow(drop, floor((1+epoch)/epochs_drop))
    return lrate

def sort_dictionary(dic):
    temp = sorted(dic.keys())
    temp_dic = {}
    
    for key in temp:
        temp_dic[key] = dic[key]
    
    return temp_dic

def main(argv):
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    # adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    augment = ImageDataGenerator(preprocessing_function=prep.preprocessing_for_training)

    device_client_dic = {}
    alg_selected_device = []

    total_demand = 50000
    num_node = 5

    for n in range(num_node):
        next_node = (n + 1) % num_node
        last_node = n - 1
        
        if last_node == -1:
            last_node = num_node - 1
            device_client_dic[n] = [next_node, last_node]
            continue
            
        device_client_dic[n] = [last_node, next_node]
            
    for num in range(num_node):
        alg_selected_device.append(total_demand/num_node)

    num_device = len(alg_selected_device)
    num_center_epoch = 1
    num_local_epoch = 5
    num_round = 10000
    center_batch_size = 64
    local_batch_size = 64

    show = 0

    device_train_data = {}
    device_test_data = {}

    for _ in range(num_round):
        print("\033[1m" + "Round: " + str(_) + '\033[0m')
        start_with = 0
        end_with = 0

        for device in device_client_dic:
            print(device)
            if(_ == 0):
                #Define an estimator model
                #Initialize every device (e.g., all devices are initialized with same parameters)
                locals()['model_{}'.format(device)] = Device_Model(device)
                locals()['model_{}'.format(device)].history['val_loss'] = [2.3840]
                locals()['model_{}'.format(device)].history['val_acc'] = [0.0976]

                locals()['model_{}'.format(device)].add_client_list(device_client_dic[device], device_client_dic)
                locals()['model_{}'.format(device)].set_weight(device_client_dic, num_node)

                temp_arange = []
                temp_amount = int(alg_selected_device[device]/10) + 10

                end_with += temp_amount
                end_with %= 5000
                temp_arange = [start_with, end_with]

                train_image_temp, train_label_temp = prep.prepare_for_training_data_selected_random_range(device, temp_arange, train_images, train_labels)
                start_with = end_with
                start_with %= 5000

                device_train_data[device] = [train_image_temp, train_label_temp]


            #Local training on each device 
            for epo in range(num_local_epoch):
                train_image_crop = np.stack([prep.random_crop(device_train_data[device][0][i], 24, 24) for i in range(len(device_train_data[device][0]))], axis=0)

                for random_ in range(10):
                    train_new_image, train_new_label = shuffle(train_image_crop, 
                                                            device_train_data[device][1], 
                                                            random_state=randint(0, train_image_crop.shape[0]))

                history_temp = locals()['model_{}'.format(device)].weight.fit_generator(
                    augment.flow(train_new_image, train_new_label, batch_size=local_batch_size), 
                    epochs=1, 
                    callbacks=[callback],
                    verbose=show)

            # Update from x^(t+1/2) to x^(t+1)   (line 4)
            locals()['model_{}'.format(device)].update_parameter()

            # Count weight_q (line 5)
            locals()['model_{}'.format(device)].count_q()

        # Update all weight_hat for each device (line 6, 7)
        for device in device_client_dic:
            for dev in device_client_dic[device]:
                locals()['model_{}'.format(device)].set_client_weight(dev, locals()['model_{}'.format(dev)].weight_q)

        # Update own weight_hat for each device (line 8)
        for device in device_client_dic:
            locals()['model_{}'.format(device)].update_own_q()

        for device in device_client_dic:
            #Evaluate with new weight
            test_d = np.stack([prep.preprocessing_for_testing(test_images[i]) for i in range(10000)], axis=0)

            test_new_image, test_new_label = shuffle(test_d, test_label, 
                                                    random_state=randint(1, train_images.shape[0]))

            history_temp = locals()['model_{}'.format(device)].weight.evaluate(test_new_image, 
                                                                            test_new_label, 
                                                                            batch_size=64,
                                                                            verbose=show)

            #Record each round accuracy and loss for every device
            locals()['model_{}'.format(device)].history['val_loss'].append(history_temp[0])
            locals()['model_{}'.format(device)].history['val_acc'].append(history_temp[1])
            print("Round: " + str(_) + ", Device:" + str(device) + ", " + "Result: ", str(history_temp[1]))

    #     if _ % 100 == 0 or _ == num_round - 1:
    #         with open('Federated_Learning_Data/15-nodes/ring_result_matrix_05.txt', 'w+') as f:
    #             for device in device_client_dic:
    #                 f.write(str(device))
    #                 f.write(str(locals()['model_{}'.format(device)].history['val_acc']))
    #                 f.write(str(locals()['model_{}'.format(device)].history['val_loss']))
    #                 f.write('\n')

if __name__ == '__main__':
    main(sys.argv)