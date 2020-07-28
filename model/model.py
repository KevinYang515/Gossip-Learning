from model.device_model import Device_Model

def init_model(device_num, device_client_dic, num_node):
    """
    Model Initialization
    :param device number which means which device we want
    :param dictionary which contains all the information of each device
    :param how many devices 
    :return the model of target device 
    """
    model_x = Device_Model(device_num)
    model_x.history['val_loss'] = [2.3840]
    model_x.history['val_acc'] = [0.0976]
    model_x.add_client_list(device_client_dic[device_num], device_client_dic)
    model_x.set_weight(device_client_dic, num_node)

    return model_x

def record_history_print(round, device, model_x, history_temp):
    """
    Record accuracy and loss result to history_total
    :param the model which history needs to be appended new result of accuracy and loss
    :param the dictionary stored the accuracy and loss for this round
    """
    record_history(model_x, history_temp)
    print("Round: " + str(round) + ", Device:" + str(device) + ", " + "Result: ", str(history_temp[1]))

def record_history(model_x, history_temp):
    """
    Record accuracy and loss result to history_total
    :param the model which history needs to be appended new result of accuracy and loss
    :param the dictionary stored the accuracy and loss for this round
    """
    model_x.history['val_loss'].append(history_temp[0])
    model_x.history['val_acc'].append(history_temp[1])

def training_once_for_gos(model_x, train_new_image, train_new_label, training_info, augment, callback):
    """
    It will make model_x train for one time.
    :param the model for training
    :param the training images for input model
    :param the training labels for input model
    :param dictionary for training detailed settings
    :param function to be ececuted before every training epoch which is mostly for data preprocessing
    :param callback function for adjusting learning rate (i.e., learning rate decay)
    :return the result after training once
    """
    history_temp = model_x.weight.fit_generator(
                        augment.flow(train_new_image, train_new_label, 
                            batch_size=training_info["local_batch_size"]),
                            epochs=1,
                            callbacks=[callback],
                            verbose=training_info["show"])
    return history_temp

def print_all_device_history(device, model_x):
    """
    Print accuracy and loss information of all devices
    :param the device we want for printing
    :param the model we want to show its accuracy and loss
    """
    print("\n================================== Device: " + str(device) + " ==================================")
    print("Accuracy: " + str(model_x.history['val_acc']))
    print("Loss: " + str(model_x.history['val_loss']))
