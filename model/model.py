from model.device_model import Device_Model

def init_model(device_num, device_client_dic, num_node):
    model_x = Device_Model(device_num)
    model_x.history['val_loss'] = [2.3840]
    model_x.history['val_acc'] = [0.0976]
    model_x.add_client_list(device_client_dic[device_num], device_client_dic)
    model_x.set_weight(device_client_dic, num_node)

    return model_x

def record_history(model_x, history_temp):
    """
    Record accuracy and loss result to history_total
    :param the model which history needs to be appended new result of accuracy and loss
    :param the dictionary stored the accuracy and loss for this round
    """
    model_x.history['val_loss'].append(history_temp[0])
    model_x.history['val_acc'].append(history_temp[1])
