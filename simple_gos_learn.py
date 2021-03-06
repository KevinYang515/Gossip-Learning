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
from data.preprocessing import prepare_for_training_data_selected_random_range, preprocessing_for_training, separate_and_preprocess_for_gos, evaluate_with_new_model_for_gos
from model.model import init_model, record_history_print, print_all_device_history, training_once_for_gos
from data.read_data import read_simple_setting
from data.relationship import generate_ring
from data.data_utils import load_cifar10_data, train_test_label_to_categorical

from math import floor
from sklearn.utils import shuffle
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
    detailed_setting = read_simple_setting()
    if (len(argv) == 2):
        detailed_setting["training_info"]["num_device"] = int(argv[1])
    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10_data()
    # Transfer train and test label to be categorical
    train_label, test_label = train_test_label_to_categorical(train_labels, test_labels)

    # adjust parameters of the model
    callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    augment = ImageDataGenerator(preprocessing_function=preprocessing_for_training)

    training_info = detailed_setting["training_info"]
    device_client_dic = generate_ring(training_info["num_device"])
    alg_selected_device = [training_info["total_demand"]/training_info["num_device"] for num in range(training_info["num_device"])]

    device_train_data, device_test_data = {}, {}

    for _ in range(training_info["num_round"]):
        print("\033[1m" + "\nRound: " + str(_) + '\033[0m')
        start_with, end_with = 0, 0

        for device in device_client_dic:
            if(_ == 0):
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
                train_new_image, train_new_label = separate_and_preprocess_for_gos(device_train_data[device])
                history_temp = training_once_for_gos(locals()['model_{}'.format(device)], train_new_image, train_new_label, training_info, augment, callback)
                
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
            history_temp = evaluate_with_new_model_for_gos(locals()['model_{}'.format(device)], training_info, test_images, test_label)
            
            #Record and print each round accuracy and loss for every device
            record_history_print(_, device, locals()['model_{}'.format(device)], history_temp)
            
    for device in device_client_dic:
        print_all_device_history(device, locals()['model_{}'.format(device)])

if __name__ == '__main__':
    main(sys.argv)