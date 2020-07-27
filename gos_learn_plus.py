from __future__ import absolute_import, division, print_function, unicode_literals
# Set Specific GPU in tensorflow
# We can check GPU information with command ```nvidia-smi```.
# If there is only one GPU avaliable, we set the os environ to "0".
# If there are two GPU avaliable and we would like to set the second GPU, 
#   we set the os environ to "1".
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import tensorflow as tf
import numpy as np
import random
from data.preprocessing import random_crop, prepare_for_training_data_selected_random_range, preprocessing_for_training, preprocessing_for_testing
from model.model import init_model, record_history_print, print_all_device_history
from data.read_data import read_setting, read_data_gossip
from data.data_utils import load_cifar10_data, train_test_label_to_categorical
from data.preprocessing import random_crop, prepare_for_training_data_selected_random, preprocessing_for_training, preprocessing_for_testing

from math import floor
from sklearn.utils import shuffle
from random import randint
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

def main(argv):
    # Read detailed settings from json file
    detailed_setting = read_setting()
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    # adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    augment = ImageDataGenerator(preprocessing_function=preprocessing_for_training)

    training_info = detailed_setting["training_info"]
    file_info = detailed_setting["file_info"]

    for f_r in range(file_info["file_value"]["start"], file_info["file_value"]["end"] + 1, file_info["file_value"]["step"]):
        for f_t in file_info["file_type"]:
            file = str(f_r) + '_' + f_t + '.txt'
            device_client_dic = read_data_gossip(file)
            alg_selected_device = [training_info["total_demand"]/training_info["num_device"] for num in range(training_info["num_device"])]
                
            device_train_data, device_test_data = {}, {}
        
            for _ in range(training_info["num_round"]):
                print("\033[1m" + "\n" + file + ", Round: " + str(_) + '\033[0m')
                start_with = 0
                end_with = 0

                for device in device_client_dic:
                    if (_ == 0):
                        #Define an estimator model and initialize every device (e.g., all devices are initialized with same parameters)
                        locals()['model_{}'.format(device)] = init_model(device, device_client_dic, training_info["num_device"])
                        
                        temp_arange = []
                        temp_amount = int(alg_selected_device[device]/10) + 10

                        end_with += temp_amount
                        end_with %= 5000
                        temp_arange = [start_with, end_with]

                        train_image_temp, train_label_temp = prepare_for_training_data_selected_random_range(device, temp_arange, train_images, train_labels)
                        start_with = end_with
                        start_with %= 5000

                        device_train_data[device] = [train_image_temp, train_label_temp]


                    #Local training on each device 
                    for epo in range(training_info["num_local_epoch"]):
                        train_image_crop = np.stack([random_crop(device_train_data[device][0][i], 24, 24) for i in range(len(device_train_data[device][0]))], axis=0)

                        for random_ in range(10):
                            train_new_image, train_new_label = shuffle(train_image_crop, 
                                                                    device_train_data[device][1], 
                                                                    random_state=randint(0, train_image_crop.shape[0]))

                        history_temp = locals()['model_{}'.format(device)].weight.fit_generator(
                                                augment.flow(train_new_image, train_new_label, 
                                                            batch_size=training_info["local_batch_size"]), 
                                                            epochs=1, 
                                                            callbacks=[callback],
                                                            verbose=training_info["show"])

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
                                                                                    batch_size=training_info["center_batch_size"],
                                                                                    verbose=training_info["show"])

                    #Record each round accuracy and loss for every device
                    record_history_print(_, device, locals()['model_{}'.format(device)], history_temp)
                        
            for device in device_client_dic:
                print_all_device_history(device, locals()['model_{}'.format(device)])

    #             if _ % 100 == 0 or _ == num_round - 1:
    #                 with open('Federated_Learning_Data/15-nodes/' + file[:-4] + '_result_matrix_05.txt', 'w+') as f:
    #                     for device in device_client_dic:
    #                         f.write(str(device))
    #                         f.write(str(locals()['model_{}'.format(device)].history['val_acc']))
    #                         f.write(str(locals()['model_{}'.format(device)].history['val_loss']))
    #                         f.write('\n')


if __name__ == '__main__':
    main(sys.argv)